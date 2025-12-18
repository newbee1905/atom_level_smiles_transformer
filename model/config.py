from dataclasses import dataclass


@dataclass
class ModelConfig:
	"""A configuration class for the Transformer model."""

	block_size: int = 1024
	vocab_size: int = 50257

	n_encoder_layer: int = 12
	n_decoder_layer: int = 12
	n_head: int = 12
	d_model: int = 768
	dropout: float = 0.1

	use_submersion: bool = False
	submersion_pooling_method: str = "mean"
	immersion_activation: str = "silu"

	use_kv_cache: bool = False
	use_gate: bool = True
	use_qk_norm: bool = True
	ffn_multiplier: int = 8 / 3

	# RoPE base frequency
	rope_theta: float = 10000.0

	layer_scale_init: float = 1e-4
