import time
import hydra
import torch
from torch.amp import autocast, GradScaler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import numpy as np
import itertools

from model.bart import Bart
from chemformer_rs.tokenizer import SMILESTokenizer
from dataset.zinc import ZincDataset

def count_parameters(model):
	"""Counts the number of trainable parameters in a model."""
	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	if num_params >= 1_000_000:
		return f"{num_params / 1_000_000:.1f}M"
	return f"{num_params / 1_000:.1f}K"


def get_gpu_memory_usage(device):
	"""Returns allocated and reserved memory in GB."""
	if device.type == "cuda":
		allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
		reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
		return allocated, reserved
	return 0.0, 0.0


def run_benchmark(cfg: DictConfig):
	"""
	Main benchmarking function with AMP (Automatic Mixed Precision) support.
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True

	dtype = getattr(torch, cfg.training.get("dtype", "float16"))
	use_amp = (dtype == torch.float16) and ("cuda" in device.type)

	tokenizer = SMILESTokenizer.from_vocab(hydra.utils.to_absolute_path(cfg.model.vocab_path))
	lmdb_path = hydra.utils.to_absolute_path(cfg.dataset.lmdb_path)
	train_indices = ZincDataset.read_split_indices(lmdb_path, "train")

	train_ds = ZincDataset(
		lmdb_path=lmdb_path,
		subset_indices=train_indices,
		tokenizer=tokenizer,
		max_length=cfg.model.max_length,
		is_training=True,
		mask_prob=cfg.dataset.mask_prob,
		span_len=cfg.dataset.span_len,
		augment_prob=cfg.dataset.augment_prob,
		span_mask_proportion=cfg.dataset.span_mask_proportion,
		span_random_proportion=cfg.dataset.span_random_proportion,
	)

	train_dl = DataLoader(
		train_ds,
		batch_size=cfg.training.batch_size,
		shuffle=True,
		num_workers=cfg.training.num_workers,
		pin_memory=True,
	)

	OmegaConf.set_struct(cfg, False)
	cfg.model.vocab_size = tokenizer.vocab_size
	cfg.model.pad_token_id = tokenizer.token_to_index("<PAD>")
	OmegaConf.set_struct(cfg, True)

	model = Bart(cfg.model).to(device)

	if cfg.training.get("compile", False):
		model = torch.compile(model)

	criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.model.pad_token_id)
	if cfg.model.get("electra_task", False):
		electra_criterion = torch.nn.BCEWithLogitsLoss()

	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.optimizer.lr)
	scaler = GradScaler(enabled=(use_amp))

	num_steps = cfg.get("bench_steps", 200)
	warmup_steps = cfg.get("warmup_steps", 20)
	total_times = []

	data_iterator = iter(train_dl)
	model.train()

	for i in range(warmup_steps + num_steps):
		if i == warmup_steps and device.type == "cuda":
			torch.cuda.reset_peak_memory_stats(device)

		t_total_start = time.time()
		try:
			batch = next(data_iterator)
		except StopIteration:
			data_iterator = iter(train_dl)
			batch = next(data_iterator)

		batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

		with autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
			gen_logits, _, disc_logits, _, _ = model(batch["src"], batch["tgt"])
			loss = criterion(gen_logits.view(-1, gen_logits.size(-1)), batch["tgt"].view(-1))
			if cfg.model.get("electra_task", False) and disc_logits is not None:
				electra_loss = electra_criterion(disc_logits, batch["electra_labels"])
				loss += electra_loss * cfg.training.electra_loss_weight

		optimizer.zero_grad()

		if use_amp:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()

		if device.type == "cuda":
			torch.cuda.synchronize()
		t_total_end = time.time()

		if i >= warmup_steps:
			total_times.append(t_total_end - t_total_start)

	throughput = 1 / np.mean(total_times)
	max_allocated, max_reserved = get_gpu_memory_usage(device)

	return throughput, max_allocated, max_reserved


@hydra.main(config_path="../config", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
	"""
	Main benchmarking function to compare Liger vs. Torch implementations.
	"""
	print("--- Liger vs. Torch Benchmark Script ---")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")

	liger_flags = ["use_liger_ff", "use_liger_norm", "use_liger_rope"]
	flag_combinations = list(itertools.product([True, False], repeat=len(liger_flags)))

	results = []

	for combo in flag_combinations:
		temp_cfg = cfg.copy()
		combo_dict = dict(zip(liger_flags, combo))

		# Create a readable name for the combination
		combo_name = []
		for flag, value in combo_dict.items():
			name = flag.split("_")[-1].upper()
			prefix = "L" if value else "T"
			combo_name.append(f"{prefix}-{name}")
		combo_name = " | ".join(combo_name)

		print(f"\n--- Benchmarking: {combo_name} ---")

		OmegaConf.set_struct(temp_cfg, False)
		for flag, value in combo_dict.items():
			temp_cfg.model[flag] = value
		OmegaConf.set_struct(temp_cfg, True)

		try:
			throughput, mem_alloc, mem_reserved = run_benchmark(temp_cfg)
			results.append(
				{
					"name": combo_name,
					"throughput": throughput,
					"mem_alloc": mem_alloc,
					"mem_reserved": mem_reserved,
					"error": None,
				}
			)
		except Exception as e:
			print(f"!!!-Error benchmarking {combo_name}: {e}")
			results.append(
				{
					"name": combo_name,
					"throughput": 0,
					"mem_alloc": 0,
					"mem_reserved": 0,
					"error": str(e),
				}
			)

	print("\n\n--- Benchmark Results Summary ---")

	# Header
	header = f"{'Combination':<30} | {'Throughput (batch/s)':<25} | {'Peak Memory (GB)':<20}"
	print(header)
	print("-" * len(header))

	results.sort(key=lambda x: x["throughput"], reverse=True)

	for res in results:
		if res["error"]:
			throughput_str = f"ERROR: {res['error'][:30]}..."
			mem_str = "N/A"
		else:
			throughput_str = f"{res['throughput']:.2f}"
			mem_str = f"{res['mem_alloc']:.2f}"
		row = f"{res['name']:<30} | {throughput_str:<25} | {mem_str:<20}"
		print(row)

	print("\nBenchmark complete.")

if __name__ == "__main__":
	main()
