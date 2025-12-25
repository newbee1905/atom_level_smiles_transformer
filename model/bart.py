import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .utils import precompute_freqs_cis
from .projection import Submersion, Immersion


class GDESEmbedding(nn.Module):
	def __init__(self, master_weight, config):
		super().__init__()
		self.master_weight = master_weight

		# E_delta: Discriminator-specific learnable features
		# Initialize small so it doesn't mask the pre-trained features early on
		self.delta_embedding = nn.Embedding(config.vocab_size, config.d_model)
		nn.init.normal_(self.delta_embedding.weight, mean=0.0, std=0.02)

	def forward(self, input_ids):
		if self.master_weight is not None:
			# Stop gradients from the Encoder's RTD loss flowing to the master weights
			shared_embeds = F.embedding(input_ids, self.master_weight).detach()

			# Add the learnable delta (Discriminator-only gradient)
			return shared_embeds + self.delta_embedding(input_ids)
		return self.delta_embedding(input_ids)


class Bart(nn.Module):
	"""The main BART-like model."""

	def __init__(self, config):
		super().__init__()
		self.config = config

		self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
		self.electra_task = getattr(config, "electra_task", False)
		self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)

		head_dim = config.d_model // config.n_head
		freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
		self.register_buffer("freqs_cos", freqs_cos, persistent=False)
		self.register_buffer("freqs_sin", freqs_sin, persistent=False)

		self.decoder_emb = nn.Embedding(config.vocab_size, config.d_model)
		if self.electra_task:
			master_weight = self.lm_head.weight if self.tie_word_embeddings else None
			self.encoder_emb = GDESEmbedding(master_weight, config)
		else:
			self.encoder_emb = nn.Embedding(config.vocab_size, config.d_model)

		if self.tie_word_embeddings:
			self.decoder_emb.weight = self.lm_head.weight
			if not self.electra_task:
				self.encoder_emb.weight = self.lm_head.weight

		self.drop = nn.Dropout(config.dropout)

		self.encoder = Encoder(config)
		self.decoder = Decoder(config)

		if self.config.use_submersion:
			self.submersion = Submersion(config)
			self.immersion = Immersion(config)

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
		encoder_emb = self.drop(self.encoder_emb(encoder_input))
		original_encoder_hidden_states, electra_logits = self.encoder(encoder_emb, self.freqs_cos, self.freqs_sin)

		reconstructed_encoder_hidden_states = None
		decoder_cross_attention_states = original_encoder_hidden_states

		# Optional Submersion/Immersion
		if self.config.use_submersion:
			latent_z = self.submersion(original_encoder_hidden_states)
			reconstructed_encoder_hidden_states = self.immersion(latent_z, encoder_input.size(1))
			decoder_cross_attention_states = reconstructed_encoder_hidden_states

		# Decoder forward
		decoder_emb = self.drop(self.decoder_emb(decoder_input))
		decoder_output, new_layer_pasts = self.decoder(
			decoder_emb,
			decoder_cross_attention_states,
			self.freqs_cos,
			self.freqs_sin,
			layer_pasts=layer_pasts,
		)

		logits = self.lm_head(decoder_output)

		return (
			logits,
			new_layer_pasts,
			electra_logits,
			original_encoder_hidden_states,
			reconstructed_encoder_hidden_states,
		)
