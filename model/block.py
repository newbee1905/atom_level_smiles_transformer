import torch
import torch.nn as nn

from .attention import MHA, DisentangledSelfAttention
from .feed_forward import FeedForward


class EncoderBlock(nn.Module):
	"""A single Transformer encoder block."""

	def __init__(self, config):
		super().__init__()
		self.attention_type = getattr(config, "encoder_attention_type", "mha")

		if config.use_liger_norm:
			from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma as RMSNorm
		else:
			from model.norm import RMSNormTorch as RMSNorm

		self.attn_norm = RMSNorm(config.d_model)
		if self.attention_type == "disentangled":
			self.attn = DisentangledSelfAttention(config)
		else:
			self.attn = MHA(config, is_decoder=False)
		self.attn_dropout = nn.Dropout(config.dropout)
		self.attn_layerscale = nn.Parameter(config.layer_scale_init * torch.ones(config.d_model))

		self.ffn_norm = RMSNorm(config.d_model)
		if config.use_liger_ff:
			self.ffn = FeedForward(config)
		else:
			from .feed_forward import FeedForwardTorch

			self.ffn = FeedForwardTorch(config)
		self.ffn_dropout = nn.Dropout(config.dropout)
		self.ffn_layerscale = nn.Parameter(config.layer_scale_init * torch.ones(config.d_model))

	def forward(self, x, freqs_cos, freqs_sin):
		attn_norm = self.attn_norm(x)
		if self.attention_type == "disentangled":
			attn_out, _ = self.attn(attn_norm)
		else:
			attn_out, _ = self.attn(attn_norm, freqs_cos, freqs_sin, is_causal=False)
		x = x + self.attn_dropout(attn_out * self.attn_layerscale)

		ffn_norm = self.ffn_norm(x)
		ffn_out = self.ffn(ffn_norm)
		x = x + self.ffn_dropout(ffn_out * self.ffn_layerscale)

		return x


class DecoderBlock(nn.Module):
	"""A single Transformer decoder block."""

	def __init__(self, config):
		super().__init__()
		if config.use_liger_norm:
			from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma as RMSNorm
		else:
			from model.norm import RMSNormTorch as RMSNorm

		self.self_attn_norm = RMSNorm(config.d_model)
		self.self_attn = MHA(config, is_decoder=True)
		self.self_attn_dropout = nn.Dropout(config.dropout)
		self.self_attn_layerscale = nn.Parameter(config.layer_scale_init * torch.ones(config.d_model))

		self.cross_attn_norm = RMSNorm(config.d_model)
		self.cross_attn = MHA(config, is_decoder=True)
		self.cross_attn_dropout = nn.Dropout(config.dropout)
		self.cross_attn_layerscale = nn.Parameter(config.layer_scale_init * torch.ones(config.d_model))

		self.ffn_norm = RMSNorm(config.d_model)
		if config.use_liger_ff:
			self.ffn = FeedForward(config)
		else:
			from .feed_forward import FeedForwardTorch

			self.ffn = FeedForwardTorch(config)
		self.ffn_dropout = nn.Dropout(config.dropout)
		self.ffn_layerscale = nn.Parameter(config.layer_scale_init * torch.ones(config.d_model))

	def forward(self, x, encoder_hidden_states, freqs_cos, freqs_sin, layer_past=None):
		# Self-attention
		self_attn_norm = self.self_attn_norm(x)
		self_attn_out, present = self.self_attn(
			self_attn_norm, freqs_cos, freqs_sin, layer_past=layer_past, is_causal=True
		)
		x = x + self.self_attn_dropout(self_attn_out * self.self_attn_layerscale)

		# Cross-attention
		cross_attn_norm = self.cross_attn_norm(x)
		cross_attn_out, _ = self.cross_attn(
			cross_attn_norm,
			freqs_cos,
			freqs_sin,
			encoder_hidden_states=encoder_hidden_states,
			is_causal=False,
		)
		x = x + self.cross_attn_dropout(cross_attn_out * self.cross_attn_layerscale)

		# FFN
		ffn_norm = self.ffn_norm(x)
		ffn_out = self.ffn(ffn_norm)
		x = x + self.ffn_dropout(ffn_out * self.ffn_layerscale)

		return x, present
