import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma as RMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

@dataclass
class ModelConfig:
	"""A configuration class for the Transformer model."""
	block_size: int = 1024
	vocab_size: int = 50257

	n_layer: int = 12
	n_head: int = 12
	d_model: int = 768
	dropout: float = 0.1

	use_kv_cache: bool = False
	use_gate: bool = True
	use_qk_norm: bool = True
	ffn_multiplier: int = 8/3

	# RoPE base frequency
	rope_theta: float = 10000.0 

	layer_scale_init: float = 1e-4

class AttributeEncoder(nn.Module):
	"""
	Fourier Feature MLP to encode float attributes (e.g., MW, LogP).
	Projects scalars to high-dim vector space.
	"""

	def __init__(self, num_props, d_model):
		super().__init__()
		# Random Gaussian matrix for Fourier projection
		self.register_buffer("B", torch.randn(num_props, d_model // 2) * 2.0 * np.pi)
		self.mlp = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.SiLU(),
			nn.Linear(d_model, d_model)
		)

	def forward(self, x):
		# x: [bsz, n_props]
		# Fourier Projection: [bsz, n_props] @ [n_props, d_model / 2] -> [bsz, d_model / 2]
		x_proj = x @ self.B 

		# Concatenate Sin and Cos -> [Batch, Dim]
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

def apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, seq_len: int):
	"""
	Applies the Rotary Positional Embedding (RoPE) to the query and key tensors.
	"""
	T = q.size(2)

	freqs_cos = freqs_cos[seq_len - T:seq_len].contiguous()
	freqs_sin = freqs_sin[seq_len - T:seq_len].contiguous()
	freqs_cos = freqs_cos.unsqueeze(0)
  freqs_sin = freqs_sin.unsqueeze(0)
	
	q_out, k_out = liger_rotary_pos_emb(
		q=q.float(),
		k=k.float(),
		cos=freqs_cos_current.to(q.dtype), 
		sin=freqs_sin_current.to(q.dtype), 
		position_ids=position_ids, 
		unsqueeze_dim=1
	)

	return q_out.type_as(q), k_out.type_as(k)

class MHA(nn.Module):
	"""
	Multi-Head Attention (MHA) with:
	- KV Cache
	- Gated Attention (G1).
	- RoPE
	"""

	def __init__(self, config: ModelConfig):
		super().__init__()
		assert config.d_model % config.n_head == 0
		self.config = config
		self.n_head = config.n_head
		self.d_model = config.d_model
		self.d_head = config.d_model // config.n_head
		self.max_seq_len = config.block_size
		self.use_kv_cache = config.use_kv_cache
		self.use_qk_norm = config.use_qk_norm
		self.use_gate = config.use_gate
		self.is_causal = True

		# Key, query, value, and gate projections
		self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
		self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
		self.resid_dropout = nn.Dropout(config.dropout)

		# G1 Gate: W_theta(X) -> Sigmoid, applied to SDPA output
		if self.use_gate:
			self.g_gate = nn.Linear(config.d_model, config.d_model, bias=False)
			print("Gated Attention (G1) is ENABLED (Sigmoid, Elementwise, Head-Specific)")

		if self.use_qk_norm:
			self.q_norm = RMSNorm(self.n_embd)
			self.k_norm = RMSNorm(self.n_embd)

	def forward(self, x, freqs_cos, freqs_sin, layer_past=None, **kwargs):
		"""
		Args:
			x (torch.Tensor): Input tensor (B, T, E).
			freqs_cos (torch.Tensor): RoPE cosine cache.
			freqs_sin (torch.Tensor): RoPE sine cache.
			layer_past (tuple, optional): (K, V) cache from previous steps.

		Returns:
			torch.Tensor: Attention output (B, T, E).
			tuple: Updated layer_past (K, V) cache.
		"""
		bsz, seq_len, d_model = x.size()

		qkv = self.c_attn(x)
		q, k, v = qkv.split(self.d_model, dim=2)
		
		# Reshape to (bsz, n_head, seq_len, d_head)
		q = q.view(bsz, seq_len, self.n_head, self.d_head).transpose(1, 2)
		k = k.view(bsz, seq_len, self.n_head, self.d_head).transpose(1, 2)
		v = v.view(bsz, seq_len, self.n_head, self.d_head).transpose(1, 2)
		
		# Get the total sequence length before concatenation
		seq_len_past = layer_past[0].size(2) if layer_past is not None else 0
		seq_len_total = seq_len_past + T

		# 2. Apply RoPE BEFORE KV Cache concatenation
		q, k = apply_rope(q, k, freqs_cos, freqs_sin, seq_len=T_total)

		# QK norm
		if self.use_qk_norm:
			q = self.q_norm(q)
			k = self.k_norm(k)

		# KV Cache Handling
		if self.use_kv_cache and layer_past is not None:
			past_k, past_v = layer_past
			k = torch.cat([past_k, k], dim=2)
			v = torch.cat([past_v, v], dim=2)
		
		# Determine the total sequence length (seq_len_kv) after caching/concatenation
		seq_len_kv = k.size(2) 
		
		# The new cache is the current k and v 
		present = (k, v) if self.use_kv_cache else None

		# Scaled Dot-Product Attention (SDPA)
		attn_mask = None
		dropout_p = self.config.dropout if self.training else 0.0

		# Attention: queries attend to keys/values autoregressively. A few cases to handle:
		if kv_cache is None or seq_len == seq_len_k:
			y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
		elif seq_len == 1:
			# During inference but with a single query in this forward pass:
			# The query has to attend to all the keys/values in the cache
			y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
		else:
			# During inference AND we have a chunk of queries in this forward pass:
			# First, each query attends to all the cached keys/values (i.e. full prefix)
			attn_mask = torch.zeros((seq_len, seq_len_k), dtype=torch.bool, device=q.device) # True = keep, False = mask
			prefix_len = seq_len_k - seq_len
			if prefix_len > 0: # can't be negative but could be zero
				attn_mask[:, :prefix_len] = True

			# Then, causal attention within this chunk
			attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
			y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
		
		# Gated Attention (G1) application
		if self.use_gate:
			# Compute the gate score: (bsz, seq_len, -1) -> (bsz, seq_len, -1)
			gate_score = torch.sigmoid(self.g_gate(x))
			
			# to (bsz, n_head, seq_len, d_head)
			gate_score = gate_score.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
			
			# multiplicative gate: Y' = Y_tilde * gate_score
			y = y * gate_score
		
		# to (bsz, seq_len, d_model)
		y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

		output = self.c_proj(y)
		output = self.resid_dropout(output)

		return output, present

class FeedForward(nn.Module):
	"""
	SwiGLU-based Feed-Forward Network (FFN).
	Implements: (input @ W_gate) * SiLU(input @ W_up) @ W_down
	"""
	def __init__(self, config: ModelConfig):
		super().__init__()
		d_model = config.d_model
		intermediate_size = int(d_model * config.ffn_multiplier)
		
		# Linear layers for SwiGLU (3 separate projections)
		self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
		self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
		self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x):
		output = self.down_proj(LigerSiLUMulFunction.apply(self.up_proj(x), self.gate_proj(x)))
		output = self.dropout(output)

		return output

class Block(nn.Module):
	"""A single Transformer block."""
	def __init__(self, config: ModelConfig):
		super().__init__()
		self.attn_norm = RMSNorm(config.d_model)
		self.attn = MHA(config)
		self.attn_dropout = nn.Dropout(config.dropout)
		self.attn_layerscale = nn.Parameter(config.layer_scale_init * torch.ones(config.d_model))

		self.ffn_norm = RMSNorm(config.d_model)
		self.ffn = FeedForward(config) 
		self.ffn_dropout = nn.Dropout(config.dropout)
		self.ffn_layerscale = nn.Parameter(config.layer_scale_init * torch.ones(config.d_model))

	def forward(self, x, freqs_cos, freqs_sin, layer_past=None):
		attn_norm = self.attn_norm(x)
		attn_out, present = self.attn(attn_norm, freqs_cos, freqs_sin, layer_past=layer_past)
		x = x + self.attn_dropout(attn_out * self.attn_layerscale)

		ffn_norm = self.ffn_norm(x)
		ffn_out = self.ffn(ffn_norm)
		x = x + self.ffn_dropout(ffn_out * self.ffn_layerscale)

		return x, present 

class Transformer(nn.Module):
	"""The main Transformer decoder model."""
	def __init__(self, config: ModelConfig):
		super().__init__()
		self.config = config

		# Precompute RoPE frequencies and register as buffers
		head_dim = config.d_model // config.n_head
		freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
		self.register_buffer("freqs_cos", freqs_cos, persistent=False)
		self.register_buffer("freqs_sin", freqs_sin, persistent=False)
		
		self.transformer = nn.ModuleDict(dict(
			emb = nn.Embedding(config.vocab_size, config.d_model),
			drop = nn.Dropout(config.dropout),
			h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
			norm = RMSNorm(config.d_model),
		))

		self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

		self.apply(self._init_weights)
		self.transformer.wte.weight = self.lm_head.weight

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			# Karpathy's recommended initialization
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
		elif isinstance(module, RMSNorm):
			nn.init.ones_(module.weight)

	def forward(self, idx, layer_pasts=None):
		bsz, seq_len = idx.size()
		
		tok_emb = self.transformer.emb(idx) 
		x = self.transformer.drop(tok_emb)

		new_layer_pasts = []
		for i, block in enumerate(self.transformer.h):
			layer_past = layer_pasts[i] if layer_pasts is not None else None
			
			# Pass precomputed RoPE caches to the block
			x, present = block(x, self.freqs_cos, self.freqs_sin, layer_past=layer_past)
			new_layer_pasts.append(present)

		x = self.transformer.norm(x)
		logits = self.lm_head(x)

		return logits, new_layer_pasts
