import os
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.bart import Bart
from chemformer_rs.tokenizer import SMILESTokenizer
from dataset.zinc import ZincDataset
from dataset.uspto_sep import UsptoSepDataset
from trainer import Trainer


def setup():
	if "WORLD_SIZE" in os.environ:
		dist.init_process_group(backend="nccl")
		rank = int(os.environ["RANK"])
		local_rank = int(os.environ["LOCAL_RANK"])
		world_size = int(os.environ["WORLD_SIZE"])
		torch.cuda.set_device(local_rank)
		device = torch.device("cuda", local_rank)
		is_ddp = True
	else:
		rank = 0
		world_size = 1
		local_rank = 0
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		is_ddp = False
	return rank, world_size, device, local_rank, is_ddp


def cleanup_ddp():
	if "WORLD_SIZE" in os.environ:
		dist.destroy_process_group()


def count_parameters(model):
	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	if num_params >= 1_000_000:
		return f"{num_params / 1_000_000:.1f}M"
	return f"{num_params / 1_000:.1f}K"


@hydra.main(config_path="config", config_name="train_config", version_base="1.3")
def main(cfg: DictConfig):
	rank, world_size, device, local_rank, is_ddp = setup()

	if rank == 0:
		print(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}")
		print(f"Running on device: {device}")

	tokenizer = SMILESTokenizer.from_vocab(hydra.utils.to_absolute_path(cfg.model.vocab_path))
	if rank == 0:
		print(f"Tokenizer loaded with {tokenizer.vocab_size} tokens.")

	lmdb_path = hydra.utils.to_absolute_path(cfg.dataset.lmdb_path)

	# Dataset selection logic
	if cfg.dataset.name == "zinc":
		DatasetClass = ZincDataset
		dataset_args = {
			"mask_prob": cfg.dataset.mask_prob,
			"span_len": cfg.dataset.span_len,
			"augment_prob": cfg.dataset.augment_prob,
			"span_mask_proportion": cfg.dataset.span_mask_proportion,
			"span_random_proportion": cfg.dataset.span_random_proportion,
		}
	elif cfg.dataset.name == "uspto_sep":
		DatasetClass = UsptoSepDataset
		dataset_args = {
			"augment_prob": cfg.dataset.augment_prob,
		}
	else:
		raise ValueError(f"Unknown dataset: {cfg.dataset.name}")

	train_indices = DatasetClass.read_split_indices(lmdb_path, "train")
	train_ds = DatasetClass(
		lmdb_path=lmdb_path,
		subset_indices=train_indices,
		tokenizer=tokenizer,
		max_length=cfg.model.max_length,
		is_training=True,
		**dataset_args,
	)
	train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
	train_dl = DataLoader(
		train_ds,
		batch_size=cfg.training.batch_size,
		sampler=train_sampler,
		shuffle=(train_sampler is None),
		num_workers=cfg.training.num_workers,
		pin_memory=True,
	)

	val_indices = DatasetClass.read_split_indices(lmdb_path, "val")
	val_ds = DatasetClass(
		lmdb_path=lmdb_path,
		subset_indices=val_indices,
		tokenizer=tokenizer,
		max_length=cfg.model.max_length,
		is_training=False,
		**dataset_args,
	)
	val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
	val_dl = DataLoader(
		val_ds,
		batch_size=cfg.training.batch_size,
		sampler=val_sampler,
		shuffle=False,
		num_workers=cfg.training.num_workers,
		pin_memory=True,
	)

	OmegaConf.set_struct(cfg, False)
	cfg.model.vocab_size = tokenizer.vocab_size
	cfg.model.pad_token_id = tokenizer.token_to_index("<PAD>")
	OmegaConf.set_struct(cfg, True)

	model = Bart(cfg.model).to(device)

	# Load pre-trained model if path is provided
	if cfg.task.get("pretrain_checkpoint"):
		if rank == 0:
			print(f"Loading pre-trained model from {cfg.task.pretrain_checkpoint}")
		checkpoint_path = hydra.utils.to_absolute_path(cfg.task.pretrain_checkpoint)

		if not os.path.exists(checkpoint_path):
			raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

		checkpoint = torch.load(checkpoint_path, map_location=device)

		model_state_dict = checkpoint.get("model_state_dict", checkpoint)

		if all(k.startswith("module.") for k in model_state_dict.keys()):
			model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}

		missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)

		if rank == 0:
			print("Pre-trained model loaded.")
			if missing_keys:
				print(f"Missing keys in state_dict: {missing_keys}")
			if unexpected_keys:
				print(f"Unexpected keys in state_dict: {unexpected_keys}")

	if rank == 0:
		print(f"Total number of trainable parameters in the BART model: {count_parameters(model)}")

	trainer = Trainer(
		cfg=cfg,
		model=model,
		train_loader=train_dl,
		val_loader=val_dl,
		device=device,
		rank=rank,
		world_size=world_size,
	)

	trainer.train()
	cleanup_ddp()


if __name__ == "__main__":
	main()
