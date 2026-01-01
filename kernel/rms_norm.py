"""
This Triton kernel is based of the LigerKernel RMSNorm but only keep Gemma-style RMSNorm,
with some slight modifycation for calculate settings to be more suitable for some local
GPU
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.libdevice import rsqrt


def _calculate_settings(n):
	"""
	Calculates block size and warp count for Triton kernels.
	This version is tuned for consumer/pro GPUs like L40S, 3090, V100, 1650.
	"""
	MAX_FUSED_SIZE = 65536
	BLOCK_SIZE = triton.next_power_of_2(n)
	if BLOCK_SIZE > MAX_FUSED_SIZE:
		raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds the max blocksize = {MAX_FUSED_SIZE}.")

	num_warps = 4
	if BLOCK_SIZE >= 8192:
		num_warps = 8
	return BLOCK_SIZE, num_warps


@triton.jit
def _row_rms_norm_fwd(
	y_ptr, x_ptr, weight_ptr, rstd_ptr,
	y_row_stride, x_row_stride, n_cols, eps,
	elementwise_affine: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
):
	row_idx = tl.program_id(0)
	col_offsets = tl.arange(0, BLOCK_SIZE)
	mask = col_offsets < n_cols

	# Load row
	x_ptr += row_idx * x_row_stride
	x_row = tl.load(x_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
	
	# Compute stats
	mean_square = tl.sum(x_row * x_row, axis=0) / n_cols
	rstd = rsqrt(mean_square + eps)
	tl.store(rstd_ptr + row_idx, rstd)

	# Normalize and apply affine transform
	normed_x = x_row * rstd
	if elementwise_affine:
		weight_row = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
		y_row = normed_x * (1.0 + weight_row)
	else:
		y_row = normed_x
	
	# Store output
	y_ptr += row_idx * y_row_stride
	tl.store(y_ptr + col_offsets, y_row, mask=mask)


@triton.jit
def _block_rms_norm_fwd(
	y_ptr, x_ptr, weight_ptr, rstd_ptr,
	y_row_stride, x_row_stride, n_rows, n_cols, eps,
	elementwise_affine: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
	BLOCK_ROW: tl.constexpr,
):
	row_block_idx = tl.program_id(0)
	row_offsets = row_block_idx * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
	col_offsets = tl.arange(0, BLOCK_SIZE)
	
	row_mask = row_offsets < n_rows
	col_mask = col_offsets < n_cols

	x_ptr += row_offsets[:, None] * x_row_stride + col_offsets[None, :]
	x_block = tl.load(x_ptr, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)

	mean_square = tl.sum(x_block * x_block, axis=1) / n_cols
	rstd = rsqrt(mean_square + eps)
	tl.store(rstd_ptr + row_offsets, rstd, mask=row_mask)
	
	normed_x = x_block * rstd[:, None]

	if elementwise_affine:
		weight_row = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0).to(tl.float32)
		y_block = normed_x * (1.0 + weight_row)[None, :]
	else:
		y_block = normed_x

	y_ptr += row_offsets[:, None] * y_row_stride + col_offsets[None, :]
	tl.store(y_ptr, y_block, mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def _row_rms_norm_bwd(
	dy_ptr, dx_ptr, x_ptr, weight_ptr, rstd_ptr, dweight_ptr,
	dy_row_stride, dx_row_stride, x_row_stride, dweight_row_stride,
	n_rows, n_cols,
	rows_per_program: tl.constexpr,
	elementwise_affine: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
):
	row_block_id = tl.program_id(0)
	row_start = row_block_id * rows_per_program
	row_end = min((row_block_id + 1) * rows_per_program, n_rows)
	col_offsets = tl.arange(0, BLOCK_SIZE)
	mask = col_offsets < n_cols

	if elementwise_affine:
		dweight_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
		weight_row = (1.0 + tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)).to(tl.float32)
	
	for row_idx in range(row_start, row_end):
		dy_row = tl.load(dy_ptr + row_idx * dy_row_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
		x_row = tl.load(x_ptr + row_idx * x_row_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
		rstd = tl.load(rstd_ptr + row_idx)

		m = dy_row * weight_row if elementwise_affine else dy_row
		
		dx_row = rstd * m
		dx_row -= rstd * (rstd * rstd / n_cols) * tl.sum(m * x_row, axis=0) * x_row
		
		if elementwise_affine:
			normed_x = x_row * rstd
			dweight_row += dy_row * normed_x
		
		tl.store(dx_ptr + row_idx * dx_row_stride + col_offsets, dx_row, mask=mask)

	if elementwise_affine:
		tl.store(dweight_ptr + row_block_id * dweight_row_stride + col_offsets, dweight_row, mask=mask)


@triton.jit
def _block_rms_norm_bwd(
	dy_ptr, dx_ptr, x_ptr, weight_ptr, rstd_ptr, dweight_ptr,
	dy_row_stride, dx_row_stride, x_row_stride, dweight_row_stride,
	n_rows, n_cols,
	elementwise_affine: tl.constexpr,
	BLOCK_SIZE: tl.constexpr,
	BLOCK_ROW: tl.constexpr,
):
	pid = tl.program_id(0)
	num_sms = tl.num_programs(0)
	col_offsets = tl.arange(0, BLOCK_SIZE)
	col_mask = col_offsets < n_cols

	if elementwise_affine:
		dweight_accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
		weight_row = (1.0 + tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0)).to(tl.float32)

	for start_row in range(pid * BLOCK_ROW, n_rows, num_sms * BLOCK_ROW):
		row_offsets = start_row + tl.arange(0, BLOCK_ROW)
		row_mask = row_offsets < n_rows
		
		dy_block = tl.load(dy_ptr + row_offsets[:, None] * dy_row_stride + col_offsets[None, :], mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
		x_block = tl.load(x_ptr + row_offsets[:, None] * x_row_stride + col_offsets[None, :], mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
		rstd_block = tl.load(rstd_ptr + row_offsets, mask=row_mask, other=0.0)

		m = dy_block * weight_row[None, :] if elementwise_affine else dy_block
		
		dx_block = rstd_block[:, None] * m
		dx_block -= (rstd_block[:, None] * rstd_block[:, None] * rstd_block[:, None] / n_cols) * tl.sum(m * x_block, axis=1)[:, None] * x_block

		if elementwise_affine:
			normed_x = x_block * rstd_block[:, None]
			dweight_accumulator += tl.sum(dy_block * normed_x, axis=0)

		tl.store(dx_ptr + row_offsets[:, None] * dx_row_stride + col_offsets[None, :], dx_block, mask=row_mask[:, None] & col_mask[None, :])
	
	if elementwise_affine:
		tl.store(dweight_ptr + pid * dweight_row_stride + col_offsets, dweight_accumulator, mask=col_mask)


class _TritonRMSNorm(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, weight, eps, elementwise_affine, in_place, row_mode):
		shape = x.shape
		x = x.reshape(-1, shape[-1])
		n_rows, n_cols = x.shape

		BLOCK_SIZE, num_warps = _calculate_settings(n_cols)

		y = torch.empty_like(x)
		rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
		
		use_row_wise_kernel = row_mode or (BLOCK_SIZE > 256 or n_rows < 4096 * 8)

		if use_row_wise_kernel:
			grid = (n_rows,)
			_row_rms_norm_fwd[grid](
				y, x, weight, rstd,
				y.stride(0), x.stride(0), n_cols, eps,
				elementwise_affine=elementwise_affine,
				BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
			)
		else:
			BLOCK_ROW = 16
			grid = (triton.cdiv(n_rows, BLOCK_ROW),)
			_block_rms_norm_fwd[grid](
				y, x, weight, rstd,
				y.stride(0), x.stride(0), n_rows, n_cols, eps,
				elementwise_affine=elementwise_affine,
				BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
				BLOCK_ROW=BLOCK_ROW,
			)

		ctx.save_for_backward(x, weight, rstd)
		ctx.elementwise_affine = elementwise_affine
		ctx.in_place = in_place
		ctx.row_mode = row_mode
		ctx.BLOCK_SIZE = BLOCK_SIZE
		ctx.num_warps = num_warps
		return y.reshape(shape)

	@staticmethod
	def backward(ctx, dy):
		x, weight, rstd = ctx.saved_tensors
		shape = dy.shape
		dy = dy.reshape(-1, shape[-1])
		n_rows, n_cols = dy.shape
		
		dx = torch.empty_like(x) if not ctx.in_place else dy

		sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
		if ctx.elementwise_affine:
			dweight = torch.zeros((sm_count, n_cols), dtype=torch.float32, device=x.device)
		else:
			dweight = None
		
		use_row_wise_kernel = ctx.row_mode or (ctx.BLOCK_SIZE > 256 or n_rows < 4096 * 8)

		if use_row_wise_kernel:
			rows_per_program = triton.cdiv(n_rows, sm_count)
			grid = (sm_count,)
			_row_rms_norm_bwd[grid](
				dy, dx, x, weight, rstd, dweight,
				dy.stride(0), dx.stride(0), x.stride(0), dweight.stride(0) if ctx.elementwise_affine else 0,
				n_rows, n_cols,
				rows_per_program=rows_per_program,
				elementwise_affine=ctx.elementwise_affine,
				BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps,
			)
		else:
			BLOCK_ROW = 64
			grid = (sm_count,)
			_block_rms_norm_bwd[grid](
				dy, dx, x, weight, rstd, dweight,
				dy.stride(0), dx.stride(0), x.stride(0), dweight.stride(0) if ctx.elementwise_affine else 0,
				n_rows, n_cols,
				elementwise_affine=ctx.elementwise_affine,
				BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps,
				BLOCK_ROW=BLOCK_ROW,
			)

		dweight_out = dweight.sum(0).to(weight.dtype) if ctx.elementwise_affine else None
		return dx.reshape(shape), dweight_out, None, None, None, None


class TritonRMSNorm(nn.Module):
	def __init__(self, hidden_size, eps=1e-6, elementwise_affine=True, in_place=True, row_mode=None):
		super().__init__()
		self.hidden_size = hidden_size
		self.eps = eps
		self.elementwise_affine = elementwise_affine
		self.in_place = in_place
		self.row_mode = row_mode

		if self.elementwise_affine:
			self.weight = nn.Parameter(torch.zeros(hidden_size))
		else:
			self.register_parameter("weight", None)

	def forward(self, x):
		return _TritonRMSNorm.apply(x, self.weight, self.eps, self.elementwise_affine, self.in_place, self.row_mode)

	def extra_repr(self):
		return (
			f"hidden_size={self.hidden_size}, eps={self.eps}, "
			f"elementwise_affine={self.elementwise_affine}, in_place={self.in_place}, "
			f"row_mode={self.row_mode} (Gemma-style with offset=1.0)"
		)
