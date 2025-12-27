import torch
from torch import Tensor
from typing import List, Dict, Any
import torch.distributed as dist


@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
	"""
	Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
	Uses a quintic iteration to maximize the slope at zero.
	"""
	assert G.ndim >= 2
	a, b, c = (3.4445, -4.7750, 2.0315)
	X = G.to(torch.float32)

	if G.size(-2) > G.size(-1):
		X = X.mT

	X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

	for _ in range(steps):
		A = X @ X.mT
		B = b * A + c * A @ A
		X = a * X + B @ X

	if G.size(-2) > G.size(-1):
		X = X.mT
	return X


@torch.compile
def adamw_update_kernel(
	p: Tensor,
	grad: Tensor,
	exp_avg: Tensor,
	exp_avg_sq: Tensor,
	lr: float,
	wd: float,
	beta1: float,
	beta2: float,
	eps: float,
	step: int,
):
	"""
	Compiled AdamW kernel for dense updates.
	"""

	if wd != 0:
		p.mul_(1 - lr * wd)

	# Decay the first and second moment running average coefficient
	exp_avg.lerp_(grad, 1 - beta1)
	exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

	# Bias Correction calculations
	bias_correction1 = 1 - beta1**step
	bias_correction2 = 1 - beta2**step

	step_size = lr / bias_correction1
	bias_correction2_sqrt = bias_correction2**0.5

	# Denom = sqrt(v_t) / sqrt(1-beta2^t) + eps
	denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

	# p = p - step_size * (m_t / denom)
	p.addcdiv_(exp_avg, denom, value=-step_size)


class Muon(torch.optim.Optimizer):
	"""
	Single-device Muon with integrated Dense AdamW support.
	"""

	def __init__(
		self,
		param_groups: List[Dict[str, Any]],
		lr=0.02,
		weight_decay=0.01,
		momentum=0.95,
		nesterov=True,
		ns_steps=5,
		adam_betas=(0.8, 0.95),
		adam_eps=1e-8,
	):
		for group in param_groups:
			assert "use_muon" in group, "Each param_group must have a 'use_muon' flag."
			if group["use_muon"]:
				group.setdefault("lr", lr)
				group.setdefault("momentum", momentum)
				group.setdefault("weight_decay", weight_decay)
				group.setdefault("nesterov", nesterov)
				group.setdefault("ns_steps", ns_steps)
			else:
				group.setdefault("lr", lr * 0.01)
				group.setdefault("betas", adam_betas)
				group.setdefault("eps", adam_eps)
				group.setdefault("weight_decay", weight_decay)

		super().__init__(param_groups, {})

	@torch.no_grad()
	def step(self, closure=None):
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			for p in group["params"]:
				if p.grad is None:
					continue

				state = self.state[p]
				if group["use_muon"]:
					if "momentum_buffer" not in state:
						state["momentum_buffer"] = torch.zeros_like(p.grad)

					buf = state["momentum_buffer"]
					buf.lerp_(p.grad, 1 - group["momentum"])
					g = p.grad.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

					original_shape = p.shape
					if p.ndim > 2:
						g = g.view(g.size(0), -1)

					g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

					if group["weight_decay"] > 0:
						p.mul_(1 - group["lr"] * group["weight_decay"])

					scale = max(1.0, g.size(-2) / g.size(-1)) ** 0.5
					p.add_(g.view(original_shape), alpha=-group["lr"] * scale)
				else:
					if "exp_avg" not in state:
						state["exp_avg"] = torch.zeros_like(p)
						state["exp_avg_sq"] = torch.zeros_like(p)
						state["step"] = 0

					state["step"] += 1
					adamw_update_kernel(
						p,
						p.grad,
						state["exp_avg"],
						state["exp_avg_sq"],
						group["lr"],
						group["weight_decay"],
						group["betas"][0],
						group["betas"][1],
						group["eps"],
						state["step"],
					)
		return loss


class DistMuon(torch.optim.Optimizer):
	"""
	Distributed Muon with integrated Dense AdamW.

	Fixes from original reference:
	1. Respects per-parameter-group Learning Rates and Weight Decays.
	   (Original collapsed all Muon params to the settings of the first group).
	2. Optimized AdamW kernel for non-Muon parameters.
	"""

	def __init__(
		self,
		param_groups,
		lr=0.02,
		weight_decay=0.01,
		momentum=0.95,
		nesterov=True,
		ns_steps=5,
		adam_betas=(0.8, 0.95),
		adam_eps=1e-8,
	):
		defaults = dict(
			lr=lr,
			weight_decay=weight_decay,
			momentum=momentum,
			nesterov=nesterov,
			ns_steps=ns_steps,
			adam_betas=adam_betas,
			adam_eps=adam_eps,
		)

		# Re-organize params to separate Muon vs AdamW, but keep track of source group settings
		adamw_groups = []
		muon_params_info = []  # Stores (param, shape, settings_dict)

		for group in param_groups:
			assert "use_muon" in group, "Each param_group must have a 'use_muon' flag."

			if group["use_muon"]:
				# Extract settings for this specific group to preserve them after reshaping
				g_settings = {
					"lr": group.get("lr", lr),
					"weight_decay": group.get("weight_decay", weight_decay),
					"momentum": group.get("momentum", momentum),
					"nesterov": group.get("nesterov", nesterov),
					"ns_steps": group.get("ns_steps", ns_steps),
				}

				for p in group["params"]:
					muon_params_info.append((p, p.shape, g_settings))
			else:
				# Handle AdamW groups
				group.setdefault("betas", adam_betas)
				group.setdefault("eps", adam_eps)
				if "lr" not in group:
					group["lr"] = lr * 0.01

				adamw_groups.append(group)

		# Bucket Muon params by shape
		# Sort by numel for deterministic ordering across ranks
		muon_params_info.sort(key=lambda x: x[0].numel(), reverse=True)

		unique_shapes = sorted(list({x[1] for x in muon_params_info}), key=lambda s: tuple(s), reverse=True)
		muon_shape_groups = []

		for shape in unique_shapes:
			# Get all params with this shape
			shape_params = [x for x in muon_params_info if x[1] == shape]

			# Construct the group
			group_params = [x[0] for x in shape_params]
			group_settings = [x[2] for x in shape_params]  # List of dicts matching params

			new_group = {
				"params": group_params,
				"use_muon": True,
				"zero_buffer": torch.zeros_like(group_params[0]),
				"per_param_settings": group_settings,  # Store individual settings
			}
			muon_shape_groups.append(new_group)

		final_groups = adamw_groups + muon_shape_groups
		super().__init__(final_groups, defaults)

	@torch.no_grad()
	def step(self, closure=None):
		rank = dist.get_rank()
		world_size = dist.get_world_size()

		# Update AdamW params (Standard Local Update)
		for group in self.param_groups:
			if group.get("use_muon", False):
				continue

			for p in group["params"]:
				if p.grad is None:
					continue
				state = self.state[p]

				if not state:
					state["step"] = 0
					state["exp_avg"] = torch.zeros_like(p)
					state["exp_avg_sq"] = torch.zeros_like(p)

				state["step"] += 1

				adamw_update_kernel(
					p,
					p.grad,
					state["exp_avg"],
					state["exp_avg_sq"],
					group["lr"],
					group["weight_decay"],
					group["betas"][0],
					group["betas"][1],
					group["eps"],
					state["step"],
				)

		# Update Muon params
		muon_groups = [g for g in self.param_groups if g.get("use_muon", False)]
		if not muon_groups:
			return None

		reduce_futures = []
		for group in muon_groups:
			params = group["params"]
			zero_buffer = group["zero_buffer"]

			# Split into chunks for each rank
			for base_i in range(0, len(params), world_size):
				param_slice = params[base_i : base_i + world_size]
				rs_input = [p.grad if p.grad is not None else zero_buffer for p in param_slice]
				rs_input.extend([zero_buffer] * (world_size - len(rs_input)))

				owner_idx = base_i + rank
				rs_output = (
					params[owner_idx].grad
					if owner_idx < len(params) and params[owner_idx].grad is not None
					else torch.empty_like(zero_buffer)
				)

				work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
				reduce_futures.append(work)

		# Compute updates on the scattered shards and kick off all-gathers
		future_idx = 0
		gather_futures = []

		for group in muon_groups:
			params = group["params"]
			settings = group["per_param_settings"]  # Retrieved the stored settings
			zero_buffer = group["zero_buffer"]

			for base_i in range(0, len(params), world_size):
				reduce_futures[future_idx].wait()
				future_idx += 1

				owner_idx = base_i + rank
				if owner_idx < len(params):
					p = params[owner_idx]
					g = p.grad  # This is now the averaged gradient from reduce_scatter

					# Retrieve specific settings (LR, WD, etc.) for this specific parameter
					p_conf = settings[owner_idx]

					if g is not None:
						state = self.state[p]
						if "momentum_buffer" not in state:
							state["momentum_buffer"] = torch.zeros_like(g)

						buf = state["momentum_buffer"]
						buf.lerp_(g, 1.0 - p_conf["momentum"])
						g = g.lerp_(buf, p_conf["momentum"]) if p_conf["nesterov"] else buf

						original_shape = p.shape
						if p.ndim > 2:
							g = g.view(g.size(0), -1)

						g = zeropower_via_newtonschulz5(g, steps=p_conf["ns_steps"])

						if p_conf["weight_decay"] > 0:
							p.mul_(1 - p_conf["lr"] * p_conf["weight_decay"])

						scale = max(1.0, g.size(-2) / g.size(-1)) ** 0.5
						p.add_(g.view(original_shape), alpha=-p_conf["lr"] * scale)

				# All-gather the updated parameters back to all ranks
				ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
				ag_output = params[base_i : base_i + world_size]
				ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))])

				work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
				gather_futures.append(work)

		torch.futures.collect_all(gather_futures).wait()
		return None
