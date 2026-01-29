import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import itertools
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path to allow for relative imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model.bart import Bart
from chemformer_rs.tokenizer import SMILESTokenizer
from dataset.zinc import ZincDataset


def get_model(cfg: DictConfig, device, seed=42):
	"""Initializes and returns a model with a specific seed."""
	# The tokenizer path is relative to the original config path
	tokenizer_path = hydra.utils.to_absolute_path(cfg.model.vocab_path)
	tokenizer = SMILESTokenizer.from_vocab(tokenizer_path)

	# We need to create a mutable copy of the config to modify it
	cfg_copy = cfg.copy()
	OmegaConf.set_struct(cfg_copy, False)
	cfg_copy.model.vocab_size = tokenizer.vocab_size
	cfg_copy.model.pad_token_id = tokenizer.token_to_index("<PAD>")
	OmegaConf.set_struct(cfg_copy, True)

	torch.manual_seed(seed)
	model = Bart(cfg_copy.model).to(device)
	model.eval()  # Set model to evaluation mode
	return model


def get_fixed_batch(cfg: DictConfig, device):
	"""Fetches a single, consistent batch of data."""
	tokenizer_path = hydra.utils.to_absolute_path(cfg.model.vocab_path)
	tokenizer = SMILESTokenizer.from_vocab(tokenizer_path)

	lmdb_path = hydra.utils.to_absolute_path(cfg.dataset.lmdb_path)

	# Use a small subset of indices for speed
	try:
		indices = ZincDataset.read_split_indices(lmdb_path, "train")[: cfg.training.batch_size]
	except (KeyError, FileNotFoundError):
		print(f"Warning: Could not find train split, creating dummy indices for {lmdb_path}.")
		indices = list(range(cfg.training.batch_size))

	dataset = ZincDataset(
		lmdb_path=lmdb_path,
		subset_indices=indices,
		tokenizer=tokenizer,
		max_length=cfg.model.max_length,
		is_training=True,
		augment_prob=0.0,  # No augmentation for consistency
		mask_prob=cfg.dataset.mask_prob,
		span_len=cfg.dataset.span_len,
		span_mask_proportion=cfg.dataset.span_mask_proportion,
		span_random_proportion=cfg.dataset.span_random_proportion,
	)

	data_loader = DataLoader(
		dataset,
		batch_size=cfg.training.batch_size,
		shuffle=False,  # No shuffling for consistency
		num_workers=0,
	)

	try:
		batch = next(iter(data_loader))
		return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
	except StopIteration:
		print("Error: The dataset is empty or could not produce a batch.")
		sys.exit(1)


def run_forward_pass(model, batch, cfg):
	"""Runs a forward pass and returns model outputs."""
	device = next(model.parameters()).device
	dtype = getattr(torch, cfg.training.get("dtype", "float16"))
	use_amp = (dtype == torch.float16) and ("cuda" in device.type)

	with torch.no_grad():
		with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
			outputs = model(batch["src"], batch["tgt"])

	gen_logits, _, disc_logits, _, _ = outputs
	return gen_logits, disc_logits


def check_for_nan_inf(tensor: torch.Tensor, name: str) -> bool:
	"""Checks a tensor for NaN or Inf values and returns True if found."""
	if tensor is None:
		return False
	has_nan = torch.isnan(tensor).any()
	has_inf = torch.isinf(tensor).any()
	if has_nan or has_inf:
		print(f"  - FAIL: Found NaN or Inf in {name}.")
		return True
	return False


@hydra.main(config_path="../config", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
	"""
	Main function to test all kernel combinations for NaN/Inf values.
	"""
	print("--- Kernel NaN/Inf Stability Test Script ---")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")

	fixed_batch = get_fixed_batch(cfg, device)

	components = ["ff", "norm", "rope"]
	kernels = ["torch", "triton", "liger"]

	kernel_combinations = list(itertools.product(kernels, repeat=len(components)))

	failed_combinations = []
	results = []

	for combo in kernel_combinations:
		temp_cfg = cfg.copy()
		combo_map = dict(zip(components, combo))
		combo_name = " | ".join([f"{c.upper()}:{k.upper():<7}" for c, k in zip(components, combo)])
		print(f"\n--- Testing: {combo_name} ---")

		config_updates = {}
		for component, kernel_type in combo_map.items():
			use_liger_flag = f"use_liger_{component}"
			use_default_liger_flag = f"use_default_liger_{component}"

			if kernel_type == "torch":  # No custom kernel
				config_updates[use_liger_flag] = False
			elif kernel_type == "triton":  # Custom kernel
				config_updates[use_liger_flag] = True
				config_updates[use_default_liger_flag] = False
			elif kernel_type == "liger":  # Liger kernel
				config_updates[use_liger_flag] = True
				config_updates[use_default_liger_flag] = True

		OmegaConf.set_struct(temp_cfg, False)
		for flag, value in config_updates.items():
			temp_cfg.model[flag] = value
		OmegaConf.set_struct(temp_cfg, True)

		status = "PASS"
		error_message = ""
		try:
			model = get_model(temp_cfg, device)
			gen_logits, disc_logits = run_forward_pass(model, fixed_batch, temp_cfg)

			gen_failed = check_for_nan_inf(gen_logits, "Generator logits")
			disc_failed = check_for_nan_inf(disc_logits, "Discriminator logits")

			if gen_failed or disc_failed:
				status = "FAIL (NaN/Inf)"
				failed_combinations.append(combo_name)

		except Exception as e:
			import traceback
			status = "FAIL (Error)"
			error_message = str(e)
			failed_combinations.append(combo_name)
			print(f"  - EXCEPTION during test: {e}")
			traceback.print_exc()

		results.append({"name": combo_name, "status": status, "error": error_message})
		if status.startswith("PASS"):
			print("  - PASS: No NaN or Inf values detected.")


	print("\n\n--- Test Summary ---")
	df = pd.DataFrame(results)
	print(df.to_string(index=False))

	if failed_combinations:
		print("\n--- Failed Combinations ---")
		for name in failed_combinations:
			print(name)
	else:
		print("\nAll kernel combinations passed successfully.")


if __name__ == "__main__":
	main()
