import torch
import torch.nn as nn
import torch.nn.functional as F

from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma as RMSNorm

from .config import ModelConfig
from .utils import apply_rope


class MHA(nn.Module):
	"""
	Multi-Head Attention (MHA) with:
	- KV Cache
	- Gated Attention (G1).
	- RoPE
	- Cross-Attention
	"""

	def __init__(self, config: ModelConfig, is_decoder: bool = False):
		super().__init__()
		assert config.d_model % config.n_head == 0
		self.config = config
		self.n_head = config.n_head
		self.d_model = config.d_model
		self.d_head = config.d_model // config.n_head
		self.max_seq_len = config.block_size
		self.use_kv_cache = config.use_kv_cache and is_decoder
		self.use_qk_norm = config.use_qk_norm
		self.use_gate = config.use_gate
		self.is_decoder = is_decoder

		# Key, query, value projections
		self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
		self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
		self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
		self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
		self.resid_dropout = nn.Dropout(config.dropout)

		if self.use_gate:
			self.g_gate = nn.Linear(config.d_model, config.d_model, bias=False)

		if self.use_qk_norm:
			self.q_norm = RMSNorm(self.d_head)
			self.k_norm = RMSNorm(self.d_head)

	def forward(
		self,
		hidden_states,
		freqs_cos,
		freqs_sin,
		layer_past=None,
		encoder_hidden_states=None,
		is_causal=False,
	):
		bsz, seq_len, _ = hidden_states.size()

		# Cross-attention
		is_cross_attention = encoder_hidden_states is not None
		if is_cross_attention:
			q = self.q_proj(hidden_states)
			k = self.k_proj(encoder_hidden_states)
			v = self.v_proj(encoder_hidden_states)
		# Self-attention
		else:
			q = self.q_proj(hidden_states)
			k = self.k_proj(hidden_states)
			v = self.v_proj(hidden_states)

		q = q.view(bsz, seq_len, self.n_head, self.d_head).transpose(1, 2)
		k = k.view(bsz, -1, self.n_head, self.d_head).transpose(1, 2)
		v = v.view(bsz, -1, self.n_head, self.d_head).transpose(1, 2)

		seq_len_past = layer_past[0].size(2) if layer_past is not None else 0
		seq_len_total = seq_len_past + seq_len

		# Apply RoPE for self-attention in decoder
		if not is_cross_attention:
			q, k = apply_rope(q, k, freqs_cos, freqs_sin, seq_len=seq_len_total)

		if self.use_qk_norm:
			q = self.q_norm(q)
			k = self.k_norm(k)

		if self.use_kv_cache and layer_past is not None:
			past_k, past_v = layer_past
			k = torch.cat([past_k, k], dim=2)
			v = torch.cat([past_v, v], dim=2)

		present = (k, v) if self.use_kv_cache else None

		seq_len_kv = k.size(2)

		y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal and seq_len == seq_len_kv)

		if self.use_gate:
			gate_score = torch.sigmoid(self.g_gate(hidden_states))
			gate_score = gate_score.view(bsz, seq_len, self.n_head, self.d_head).transpose(1, 2)
			y = y * gate_score

		y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

		output = self.c_proj(y)
		output = self.resid_dropout(output)

		return output, present
