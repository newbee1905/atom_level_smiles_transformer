"""
This Triton kernel provides a highly optimized implementation of the Rotary Positional Embedding (RoPE) operation.
It is designed for high performance and includes autotuning configurations for a wide range of GPUs.

Compared to earlier implementations inspired by liger-kernel, this version incorporates several key enhancements:
- Reduced Memory CopiesIt avoids unnecessary `.contiguous()` calls by accepting tensor strides directly in the kernel. This zero-copy or reduced-copy approach significantly lowers overhead.
- Flexible Tensor LayoutsThe kernel automatically detects whether input tensors are in `(B, S, H, D)` or `(B, H, S, D)` format, making it more versatile and easier to integrate.
"""

import torch
import triton
import triton.language as tl

from .config import get_rope_autotune_configs


@triton.autotune(
	configs=get_rope_autotune_configs(),
	key=["n_qh", "n_kh", "hd"],
)
@triton.jit
def _triton_rope_kernel(
	q_ptr,
	k_ptr,
	cos_ptr,
	sin_ptr,
	q_stride_b,
	q_stride_s,
	q_stride_h,
	k_stride_b,
	k_stride_s,
	k_stride_h,
	cos_stride_b,
	cos_stride_s,
	sin_stride_b,
	sin_stride_s,
	# Dimensions
	seq_len,
	n_qh: tl.constexpr,  # Num Query Heads
	n_kh: tl.constexpr,  # Num Key Heads
	hd: tl.constexpr,  # Head Dimension
	# Autotune-injected / Constexpr
	BLOCK_SIZE: tl.constexpr,
	BACKWARD_PASS: tl.constexpr,
):
	pid = tl.program_id(0)

	batch_idx = pid // seq_len
	seq_idx = pid % seq_len

	# Calculate Base Pointers for this Token
	q_token_ptr = q_ptr + (batch_idx * q_stride_b) + (seq_idx * q_stride_s)
	k_token_ptr = k_ptr + (batch_idx * k_stride_b) + (seq_idx * k_stride_s)

	# Handle broadcasting if cos/sin batch stride is 0 (shared freqs)
	cos_token_ptr = cos_ptr + (batch_idx * cos_stride_b) + (seq_idx * cos_stride_s)
	sin_token_ptr = sin_ptr + (batch_idx * sin_stride_b) + (seq_idx * sin_stride_s)

	# Load Cos/Sin (Shared across all heads for this token)
	offsets_dim = tl.arange(0, hd // 2)
	mask_dim = offsets_dim < (hd // 2)

	cos_val = tl.load(cos_token_ptr + offsets_dim, mask=mask_dim, other=0.0).to(tl.float32)
	sin_val = tl.load(sin_token_ptr + offsets_dim, mask=mask_dim, other=0.0).to(tl.float32)

	# Iterate over Heads in blocks
	max_heads = tl.maximum(n_qh, n_kh)
	for h_block_start in range(0, max_heads, BLOCK_SIZE):
		h_offsets = h_block_start + tl.arange(0, BLOCK_SIZE)

		# Process Query (Q)
		mask_qh = h_offsets < n_qh
		if tl.sum(mask_qh) > 0:
			off_q1 = (h_offsets[:, None] * q_stride_h) + offsets_dim[None, :]
			off_q2 = off_q1 + (hd // 2)

			q1 = tl.load(q_token_ptr + off_q1, mask=mask_qh[:, None] & mask_dim[None, :], other=0.0).to(tl.float32)
			q2 = tl.load(q_token_ptr + off_q2, mask=mask_qh[:, None] & mask_dim[None, :], other=0.0).to(tl.float32)

			if BACKWARD_PASS:
				new_q1 = q1 * cos_val[None, :] + q2 * sin_val[None, :]
				new_q2 = q2 * cos_val[None, :] - q1 * sin_val[None, :]
			else:
				new_q1 = q1 * cos_val[None, :] - q2 * sin_val[None, :]
				new_q2 = q2 * cos_val[None, :] + q1 * sin_val[None, :]

			# Store back
			tl.store(q_token_ptr + off_q1, new_q1, mask=mask_qh[:, None] & mask_dim[None, :])
			tl.store(q_token_ptr + off_q2, new_q2, mask=mask_qh[:, None] & mask_dim[None, :])

		# Process Key (K)
		mask_kh = h_offsets < n_kh
		if tl.sum(mask_kh) > 0:
			off_k1 = (h_offsets[:, None] * k_stride_h) + offsets_dim[None, :]
			off_k2 = off_k1 + (hd // 2)

			k1 = tl.load(k_token_ptr + off_k1, mask=mask_kh[:, None] & mask_dim[None, :], other=0.0).to(tl.float32)
			k2 = tl.load(k_token_ptr + off_k2, mask=mask_kh[:, None] & mask_dim[None, :], other=0.0).to(tl.float32)

			if BACKWARD_PASS:
				new_k1 = k1 * cos_val[None, :] + k2 * sin_val[None, :]
				new_k2 = k2 * cos_val[None, :] - k1 * sin_val[None, :]
			else:
				new_k1 = k1 * cos_val[None, :] - k2 * sin_val[None, :]
				new_k2 = k2 * cos_val[None, :] + k1 * sin_val[None, :]

			tl.store(k_token_ptr + off_k1, new_k1, mask=mask_kh[:, None] & mask_dim[None, :])
			tl.store(k_token_ptr + off_k2, new_k2, mask=mask_kh[:, None] & mask_dim[None, :])


class TritonRoPEFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, q, k, cos, sin):
		# The inner dimension D must be contiguous for high performance.
		if q.stride(-1) != 1:
			q = q.contiguous()
		if k.stride(-1) != 1:
			k = k.contiguous()
		if cos.stride(-1) != 1:
			cos = cos.contiguous()
		if sin.stride(-1) != 1:
			sin = sin.contiguous()

		# Extract Shapes & Determine Layout
		if q.ndim != 4:
			raise ValueError("RoPE requires 4D input tensors.")

		# Auto-detect layout by comparing seq len dim with cos/sin shape
		if cos.shape[-2] == q.shape[1]:  # [B, S, H, D]
			seq_len_dim_idx, head_dim_idx = 1, 2
		elif cos.shape[-2] == q.shape[2]:  # [B, H, S, D]
			seq_len_dim_idx, head_dim_idx = 2, 1
		else:
			raise ValueError("Could not determine sequence length dimension from input shapes.")

		bsz = q.shape[0]
		seq_len = q.shape[seq_len_dim_idx]
		n_qh = q.shape[head_dim_idx]
		n_kh = k.shape[head_dim_idx]
		hd = q.shape[3]

		# Strides
		q_s_b, q_s_s, q_s_h = q.stride(0), q.stride(seq_len_dim_idx), q.stride(head_dim_idx)
		k_s_b, k_s_s, k_s_h = k.stride(0), k.stride(seq_len_dim_idx), k.stride(head_dim_idx)

		if cos.ndim == 2:  # [S, D] -> broadcast over batch
			cos_s_b, cos_s_s = 0, cos.stride(0)
			sin_s_b, sin_s_s = 0, sin.stride(0)
		else:  # [B, S, D]
			cos_s_b, cos_s_s = cos.stride(0), cos.stride(1)
			sin_s_b, sin_s_s = sin.stride(0), sin.stride(1)

		grid = (bsz * seq_len,)
		_triton_rope_kernel[grid](
			q,
			k,
			cos,
			sin,
			q_s_b,
			q_s_s,
			q_s_h,
			k_s_b,
			k_s_s,
			k_s_h,
			cos_s_b,
			cos_s_s,
			sin_s_b,
			sin_s_s,
			seq_len,
			n_qh=n_qh,
			n_kh=n_kh,
			hd=hd,
			BACKWARD_PASS=False,
		)

		ctx.save_for_backward(cos, sin)
		ctx.params = (seq_len_dim_idx, head_dim_idx, n_qh, n_kh, hd, seq_len)
		return q, k

	@staticmethod
	def backward(ctx, dq, dk):
		cos, sin = ctx.saved_tensors
		seq_len_dim_idx, head_dim_idx, n_qh, n_kh, hd, seq_len = ctx.params

		if dq.stride(-1) != 1:
			dq = dq.contiguous()
		if dk.stride(-1) != 1:
			dk = dk.contiguous()

		bsz = dq.shape[0]

		q_s_b, q_s_s, q_s_h = dq.stride(0), dq.stride(seq_len_dim_idx), dq.stride(head_dim_idx)
		k_s_b, k_s_s, k_s_h = dk.stride(0), dk.stride(seq_len_dim_idx), dk.stride(head_dim_idx)

		if cos.ndim == 2:
			cos_s_b, cos_s_s = 0, cos.stride(0)
			sin_s_b, sin_s_s = 0, sin.stride(0)
		else:
			cos_s_b, cos_s_s = cos.stride(0), cos.stride(1)
			sin_s_b, sin_s_s = sin.stride(0), sin.stride(1)

		grid = (bsz * seq_len,)

		_triton_rope_kernel[grid](
			dq,
			dk,
			cos,
			sin,
			q_s_b,
			q_s_s,
			q_s_h,
			k_s_b,
			k_s_s,
			k_s_h,
			cos_s_b,
			cos_s_s,
			sin_s_b,
			sin_s_s,
			seq_len,
			n_qh=n_qh,
			n_kh=n_kh,
			hd=hd,
			BACKWARD_PASS=True,
		)

		return dq, dk, None, None
