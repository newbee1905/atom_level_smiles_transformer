import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma as RMSNorm


class Submersion(nn.Module):
	"""
	Projects the encoder output sequence to a single latent vector.
	The implementation is based on a plausible interpretation of the SMI-TED paper,
	as the paper's description and formulas are ambiguous.
	"""

	def __init__(self, config: ModelConfig):
		super().__init__()
		self.config = config
		self.w1 = nn.Linear(config.d_model, config.d_model)
		self.w2 = nn.Linear(config.d_model, config.d_model)
		self.norm = RMSNorm(config.d_model)

		if self.config.submersion_pooling_method == "attention":
			self.attention_net = nn.Sequential(
				nn.Linear(config.d_model, config.d_model // 2),
				nn.Tanh(),
				nn.Linear(config.d_model // 2, 1, bias=False),
			)

	def forward(self, x):
		# x is (batch, seq_len, d_model)
		if self.config.submersion_pooling_method == "mean":
			x_pooled = x.mean(dim=1)
		elif self.config.submersion_pooling_method == "attention":
			scores = self.attention_net(x)  # (B, S, 1)
			weights = F.softmax(scores, dim=1)  # (B, S, 1)
			x_pooled = torch.sum(x * weights, dim=1)  # (B, E)
		else:
			raise ValueError(f"Unknown pooling method: {self.config.submersion_pooling_method}")

		# Then, a two-layer MLP as described in the paper's spirit.
		z = self.w2(F.gelu(self.norm(self.w1(x_pooled))))
		return z


class Immersion(nn.Module):
	"""
	Projects the latent vector back to a sequence for the decoder.
	The implementation is based on a plausible interpretation of the SMI-TED paper.
	"""

	def __init__(self, config: ModelConfig):
		super().__init__()
		self.config = config
		self.w3 = nn.Linear(config.d_model, config.d_model)
		self.w4 = nn.Linear(config.d_model, config.d_model)
		self.norm = RMSNorm(config.d_model)

		if self.config.immersion_activation == "gelu":
			self.activation = F.gelu
		elif self.config.immersion_activation == "silu":
			self.activation = F.silu
		else:
			raise ValueError(f"Unknown activation function: {self.config.immersion_activation}")

	def forward(self, z, seq_len):
		# z is (batch, d_model)
		# We use a two-layer MLP to project the latent vector.
		h = self.w4(self.activation(self.norm(self.w3(z))))
		# Then, we expand it to the asequence length required by the decoder.
		h = h.unsqueeze(1).repeat(1, seq_len, 1)
		return h
