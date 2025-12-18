import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .utils import precompute_freqs_cis
from .projection import Submersion, Immersion


class Bart(nn.Module):
	"""The main BART-like model."""

	def __init__(self, config):
		super().__init__()
		self.config = config

		head_dim = config.d_model // config.n_head
		freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
		self.register_buffer("freqs_cos", freqs_cos, persistent=False)
		self.register_buffer("freqs_sin", freqs_sin, persistent=False)

		self.emb = nn.Embedding(config.vocab_size, config.d_model)
		self.drop = nn.Dropout(config.dropout)

		self.encoder = Encoder(config)
		self.decoder = Decoder(config)

		if self.config.use_submersion:
			self.submersion = Submersion(config)
			self.immersion = Immersion(config)

		self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
		self.emb.weight = self.lm_head.weight

		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, encoder_input, decoder_input, layer_pasts=None):
		# Encoder forward
		encoder_emb = self.drop(self.emb(encoder_input))
		encoder_hidden_states = self.encoder(encoder_emb, self.freqs_cos, self.freqs_sin)

		# Optional Submersion/Immersion
		if self.config.use_submersion:
			latent_z = self.submersion(encoder_hidden_states)
			encoder_hidden_states = self.immersion(latent_z, encoder_input.size(1))

		# Decoder forward
		decoder_emb = self.drop(self.emb(decoder_input))
		decoder_output, new_layer_pasts = self.decoder(
			decoder_emb,
			encoder_hidden_states,
			self.freqs_cos,
			self.freqs_sin,
			layer_pasts=layer_pasts,
		)

		logits = self.lm_head(decoder_output)

		return logits, new_layer_pasts
