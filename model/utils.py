import torch
import torch.nn as nn
import numpy as np

from liger_kernel.transformers.rope import liger_rotary_pos_emb


class AttributeEncoder(nn.Module):
	"""
	Fourier Feature MLP to encode float attributes (e.g., MW, LogP).
	Projects scalars to high-dim vector space.
	"""

	def __init__(self, num_props, d_model):
		super().__init__()
		# Random Gaussian matrix for Fourier projection
		self.register_buffer("B", torch.randn(num_props, d_model // 2) * 2.0 * np.pi)
		self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

	def forward(self, x):
		# x: [bsz, n_props]
		# Fourier Projection: [bsz, n_props] @ [n_props, d_model / 2] -> [bsz, d_model / 2]
		x_proj = x @ self.B

		# Concatenate Sin and Cos -> [bsz, d_model]
		x_embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

		# MLP and reshape to [bsz, 1, d_model] for attention
		return self.mlp(x_embed).unsqueeze(1)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
	"""
	Precomputes the cosine and sine components for Rotary Positional Embedding (RoPE).
	Returns freqs_cos and freqs_sin, both of shape (end, dim // 2).
	"""
	# Calculation of theta_i: freqs = 1 / (theta^(2i/dim))
	freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

	t = torch.arange(end, device=freqs.device)
	freqs = torch.outer(t, freqs).float()

	# cos(m * theta), sin(m * theta)
	return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
	q: torch.Tensor,
	k: torch.Tensor,
	freqs_cos: torch.Tensor,
	freqs_sin: torch.Tensor,
	seq_len: int,
):
	"""
	Applies the Rotary Positional Embedding (RoPE) to the query and key tensors.
	"""
	T = q.size(2)

	freqs_cos = freqs_cos[seq_len - T : seq_len].contiguous()
	freqs_sin = freqs_sin[seq_len - T : seq_len].contiguous()
	freqs_cos = freqs_cos.unsqueeze(0)
	freqs_sin = freqs_sin.unsqueeze(0)

	q_out, k_out = liger_rotary_pos_emb(q=q, k=k, cos=freqs_cos.to(q.dtype), sin=freqs_sin.to(q.dtype))

	return q_out.type_as(q), k_out.type_as(k)
