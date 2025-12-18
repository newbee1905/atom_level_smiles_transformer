import torch.nn as nn

from .config import ModelConfig
from .block import EncoderBlock
from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma as RMSNorm


class Encoder(nn.Module):
	def __init__(self, config: ModelConfig):
		super().__init__()
		self.config = config

		self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_encoder_layer)])
		self.norm = RMSNorm(config.d_model)

	def forward(self, x, freqs_cos, freqs_sin):
		for layer in self.layers:
			x = layer(x, freqs_cos, freqs_sin)

		return self.norm(x)
