import torch
from torch import Tensor
from typing import List, Dict, Any


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
	Optimized for multi-GPU training with communication overlap.
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
		for group in param_groups:
			assert "use_muon" in group, "Each param_group must have a 'use_muon' flag."
			if group["use_muon"]:
				group.setdefault("lr", lr)
				group.setdefault("momentum", momentum)
				group.setdefault("weight_decay", weight_decay)
				group.setdefault("nesterov", nesterov)
				group.setdefault("ns_steps", ns_steps)

				# Sort Muon params by size for deterministic block-cyclic distribution
				group["params"] = sorted(group["params"], key=lambda x: x.numel(), reverse=True)
			else:
				group.setdefault("lr", lr * 0.01)
				group.setdefault("betas", adam_betas)
				group.setdefault("eps", adam_eps)
				group.setdefault("weight_decay", weight_decay)

		super().__init__(param_groups, {})

		self.zero_buffers = {}
		for group in self.param_groups:
			if group["use_muon"]:
				for p in group["params"]:
					if p.shape not in self.zero_buffers:
						self.zero_buffers[p.shape] = torch.zeros_like(p)

	@torch.no_grad()
	def step(self, closure=None):
		rank = dist.get_rank()
		world_size = dist.get_world_size()

		# Update non-Muon parameters (Dense AdamW)
		for group in self.param_groups:
			if group["use_muon"]:
				continue

			for p in group["params"]:
				if p.grad is None:
					continue

				state = self.state[p]
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

		# Update Muon parameters (Distributed)
		muon_group = self.param_groups[0]
		params = muon_group["params"]
		if not params:
			return None

		reduce_futures = []
		for base_i in range(0, len(params), world_size):
			owner_idx = base_i + rank
			rs_input = [p.grad for p in params[base_i : base_i + world_size]]
			if len(rs_input) < world_size:
				rs_input.extend([torch.zeros_like(rs_input[0]) for _ in range(world_size - len(rs_input))])

			rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(rs_input[0])
			work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
			reduce_futures.append(work)

		gather_futures = []
		for i, base_i in enumerate(range(0, len(params), world_size)):
			reduce_futures[i].wait()
			owner_idx = base_i + rank

			if owner_idx < len(params):
				p = params[owner_idx]
				state = self.state[p]
				if "momentum_buffer" not in state:
					state["momentum_buffer"] = torch.zeros_like(p)

				buf = state["momentum_buffer"]
				buf.lerp_(p.grad, 1.0 - muon_group["momentum"])
				g = p.grad.lerp_(buf, muon_group["momentum"]) if muon_group["nesterov"] else buf
				g = zeropower_via_newtonschulz5(g, steps=muon_group["ns_steps"])

				if muon_group["weight_decay"] > 0:
					p.mul_(1 - muon_group["lr"] * muon_group["weight_decay"])

				scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
				p.add_(g, alpha=-muon_group["lr"] * scale)

			ag_input = params[owner_idx] if owner_idx < len(params) else self.zero_buffers[params[base_i].shape]
			ag_output = params[base_i : base_i + world_size]
			if len(ag_output) < world_size:
				ag_output = list(ag_output)
				ag_output.extend([torch.empty_like(ag_input) for _ in range(world_size - len(ag_output))])

			work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
			gather_futures.append(work)

		torch.futures.collect_all(gather_futures).wait()
		return None
