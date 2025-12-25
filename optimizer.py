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
	X = G.to(torch.bfloat16)

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

	exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
	exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)

	bias_correction1 = 1 - beta1**step
	bias_correction2 = 1 - beta2**step

	denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
	p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)


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
	The Muon portion is adapted from the reference implementation in muon.py,
	regrouping parameters by shape for cleaner communication logic.
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
		muon_params = []
		adamw_groups = []
		muon_defaults = {}

		# Separate parameter groups and establish defaults
		for group in param_groups:
			assert "use_muon" in group, "Each param_group must have a 'use_muon' flag."
			if group["use_muon"]:
				muon_params.extend(group["params"])
				if not muon_defaults:
					muon_defaults["lr"] = group.setdefault("lr", lr)
					muon_defaults["momentum"] = group.setdefault("momentum", momentum)
					muon_defaults["weight_decay"] = group.setdefault("weight_decay", weight_decay)
					muon_defaults["nesterov"] = group.setdefault("nesterov", nesterov)
					muon_defaults["ns_steps"] = group.setdefault("ns_steps", ns_steps)
			else:
				group.setdefault("lr", lr * 0.01)
				group.setdefault("betas", adam_betas)
				group.setdefault("eps", adam_eps)
				adamw_groups.append(group)

		# Regroup Muon parameters by shape, as in the original muon.py
		muon_shape_groups = []
		if muon_params:
			# Sort by numel for deterministic group ordering
			muon_params.sort(key=lambda p: p.numel(), reverse=True)
			shapes = sorted(list({p.shape for p in muon_params}), key=lambda s: tuple(s), reverse=True)

			for shape in shapes:
				group_params = [p for p in muon_params if p.shape == shape]
				new_group = {
					"params": group_params,
					"use_muon": True,
					"zero_buffer": torch.zeros_like(group_params[0]),
					**muon_defaults,
				}
				muon_shape_groups.append(new_group)

		final_groups = adamw_groups + muon_shape_groups
		super().__init__(final_groups, {})

	@torch.no_grad()
	def step(self, closure=None):
		rank = dist.get_rank()
		world_size = dist.get_world_size()

		# Update non-Muon parameters (Dense AdamW)
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

		# Update Muon parameters
		muon_groups = [g for g in self.param_groups if g.get("use_muon", False)]
		if not muon_groups:
			return None

		# Kick off all reduce-scatters
		reduce_futures = []
		for group in muon_groups:
			params = group["params"]
			zero_buffer = group["zero_buffer"]
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

		# Compute updates and kick off all-gathers
		future_idx = 0
		gather_futures = []
		for group in muon_groups:
			params = group["params"]
			zero_buffer = group["zero_buffer"]
			for base_i in range(0, len(params), world_size):
				reduce_futures[future_idx].wait()
				future_idx += 1

				owner_idx = base_i + rank
				if owner_idx < len(params):
					p = params[owner_idx]
					g = p.grad

					if g is not None:
						state = self.state[p]
						if "momentum_buffer" not in state:
							state["momentum_buffer"] = torch.zeros_like(g)

						buf: Tensor = state["momentum_buffer"]
						buf.lerp_(g, 1.0 - group["momentum"])
						g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

						original_shape = p.shape
						if p.ndim > 2:
							g = g.view(g.size(0), -1)

						g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

						if group["weight_decay"] > 0:
							p.mul_(1 - group["lr"] * group["weight_decay"])

						scale = max(1.0, g.size(-2) / g.size(-1)) ** 0.5
						p.add_(g.view(original_shape), alpha=-group["lr"] * scale)

				# All-gather updated parameters
				ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
				ag_output = params[base_i : base_i + world_size]
				ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))])
				work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
				gather_futures.append(work)

		torch.futures.collect_all(gather_futures).wait()
		return None
