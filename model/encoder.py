import torch.nn as nn

from .block import EncoderBlock
from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma as RMSNorm


class Encoder(nn.Module):
	"""The encoder part of the transformer."""

	def __init__(self, config):
		super().__init__()
		self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_encoder_layer)])
		self.norm = RMSNorm(config.d_model)

	def forward(self, x, freqs_cos, freqs_sin):
		for layer in self.layers:
			x = layer(x, freqs_cos, freqs_sin)

		return self.norm(x)
