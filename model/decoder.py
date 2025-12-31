import torch.nn as nn

from .block import DecoderBlock
from .norm import get_norm_class


class Decoder(nn.Module):
	"""The decoder part of the transformer."""

	def __init__(self, config):
		super().__init__()
		self.config = config
		self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_decoder_layer)])

		RMSNorm = get_norm_class(config)
		self.norm = RMSNorm(config.d_model)

	def forward(self, x, encoder_hidden_states, freqs_cos, freqs_sin, layer_pasts=None):
		new_layer_pasts = [] if self.config.use_kv_cache else None

		for i, layer in enumerate(self.layers):
			layer_past = layer_pasts[i] if layer_pasts is not None else None
			x, present = layer(x, encoder_hidden_states, freqs_cos, freqs_sin, layer_past=layer_past)

			if self.config.use_kv_cache:
				new_layer_pasts.append(present)

		return self.norm(x), new_layer_pasts
