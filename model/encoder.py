import torch.nn as nn

from .block import EncoderBlock
from liger_kernel.ops.swiglu import LigerSiLUMulFunction


class ElectraPredictionHead(nn.Module):
	"""ELECTRA-style prediction head for replaced token detection."""

	def __init__(self, config):
		super().__init__()
		self.dense = nn.Linear(config.d_model, config.d_model)
		self.activation = nn.GELU()
		self.out_proj = nn.Linear(config.d_model, 1)

	def forward(self, x):
		x = self.dense(x)
		x = self.activation(x)
		logits = self.out_proj(x).squeeze(-1)
		return logits


class Encoder(nn.Module):
	"""The encoder part of the transformer."""

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_encoder_layer)])
		if config.use_liger_norm:
			from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma as RMSNorm
		else:
			from model.norm import RMSNormTorch as RMSNorm
		self.norm = RMSNorm(config.d_model)
		self.electra_task = getattr(config, "electra_task", False)

		if self.electra_task:
			self.electra_head = ElectraPredictionHead(config)

	def forward(self, x, freqs_cos, freqs_sin):
		for layer in self.layers:
			x = layer(x, freqs_cos, freqs_sin)

		hidden_states = self.norm(x)

		if self.electra_task:
			logits = self.electra_head(hidden_states)
			return hidden_states, logits

		return hidden_states, None
