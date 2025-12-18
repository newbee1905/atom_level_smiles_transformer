import pickle

import lmdb
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset


class ZincDataset(Dataset):
	def __init__(
		self,
		lmdb_path,
		tokenizer,
		max_length,
		subset_indices=None,
		mask_prob=0.15,
		span_len=3,
		augment_prob=0.5,
		is_training=False,
	):
		self.lmdb_path = lmdb_path
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.mask_prob = mask_prob
		self.span_len = span_len
		self.augment_prob = augment_prob
		self.is_training = is_training
		self.indices = subset_indices
		self.env = None

		# Get special token IDs
		self.mask_token_id = self.tokenizer.token_to_index("<MASK>")
		self.pad_token_id = self.tokenizer.token_to_index("<PAD>")
		self.bos_token_id = self.tokenizer.token_to_index("<BOS>")
		self.eos_token_id = self.tokenizer.token_to_index("<EOS>")

		env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
		with env.begin(write=False) as txn:
			self.full_length = int(txn.get(b"__len__").decode("ascii"))
		env.close()

		if self.indices is None:
			self.indices = list(range(self.full_length))

	def _init_db(self):
		if self.env is None:
			self.env = lmdb.open(
				self.lmdb_path,
				readonly=True,
				lock=False,
				readahead=not self.is_training,
				meminit=False,
			)

	def __len__(self):
		return len(self.indices)

	@classmethod
	def read_split_indices(cls, lmdb_path, split_name):
		env = lmdb.open(lmdb_path, readonly=True, lock=False)
		with env.begin(write=False) as txn:
			split_indices = pickle.loads(txn.get(f"split_{split_name}".encode("ascii")))
		return split_indices

	def randomize_smiles(self, smiles):
		"""Default RDKit randomization."""
		mol = Chem.MolFromSmiles(smiles)
		return Chem.MolToSmiles(mol, doRandom=True, canonical=False) if mol else smiles

	def apply_span_masking(self, token_ids: np.ndarray):
		"""Replaces contiguous spans of tokens with <mask>."""
		n = len(token_ids)
		if n == 0:
			return token_ids, np.array([], dtype=bool)

		num_to_mask = max(1, int(n * self.mask_prob))

		mask = np.zeros(n, dtype=bool)
		num_masked = 0

		while num_masked < num_to_mask:
			start = np.random.randint(0, n)
			length = np.random.randint(1, self.span_len + 1)
			end = min(start + length, n)
			mask[start:end] = True
			num_masked = np.sum(mask)

		# If we've masked too much, randomly unmask some tokens
		masked_indices_locs = np.where(mask)[0]
		if len(masked_indices_locs) > num_to_mask:
			to_unmask = np.random.choice(
				masked_indices_locs,
				size=len(masked_indices_locs) - num_to_mask,
				replace=False,
			)
			mask[to_unmask] = False

		masked_token_ids = np.copy(token_ids)
		masked_token_ids[mask] = self.mask_token_id

		return masked_token_ids, mask

	def __getitem__(self, index):
		if self.env is None:
			self._init_db()

		db_index = self.indices[index]
		with self.env.begin(write=False) as txn:
			smi = txn.get(f"{db_index}".encode("ascii")).decode("ascii")

		augmented_smiles = (
			(self.randomize_smiles(smi) if np.random.rand() < self.augment_prob else smi)
			if self.is_training
			else smi
		)

		# Tokenize, but without special tokens or padding yet
		core_token_ids, _ = self.tokenizer.encode(
			augmented_smiles,
			add_bos=False,
			add_eos=False,
			pad_to_length=False,
			max_length=self.max_length - 2,  # Leave space for BOS/EOS
		)
		core_token_ids = np.array(core_token_ids)

		# Apply span masking to create the source sequence
		masked_core_token_ids, span_mask_unpadded = self.apply_span_masking(core_token_ids)

		# Prepare src and tgt sequences with special tokens and padding
		def prepare_sequence(ids, pad_id):
			ids = np.concatenate([[self.bos_token_id], ids, [self.eos_token_id]]).astype(int)
			seq_len = len(ids)
			padding_len = self.max_length - seq_len

			padded_ids = np.pad(ids, (0, padding_len), "constant", constant_values=pad_id)
			attention_mask = np.pad(
				np.ones(seq_len, dtype=int),
				(0, padding_len),
				"constant",
				constant_values=0,
			)
			return padded_ids, attention_mask

		src_ids, src_attention_mask = prepare_sequence(masked_core_token_ids, self.pad_token_id)
		tgt_ids, tgt_attention_mask = prepare_sequence(core_token_ids, self.pad_token_id)

		# Pad the span mask, adding False for special tokens and padding
		padding_len = self.max_length - len(masked_core_token_ids) - 2
		span_mask = np.pad(
			span_mask_unpadded,
			(1, 1 + padding_len),
			"constant",
			constant_values=False,
		)

		return {
			"src": torch.from_numpy(src_ids),
			"src_attention_mask": torch.from_numpy(src_attention_mask),
			"tgt": torch.from_numpy(tgt_ids),
			"tgt_attention_mask": torch.from_numpy(tgt_attention_mask),
			"span_mask": torch.from_numpy(span_mask),
		}


if __name__ == "__main__":
	from chemformer_rs.tokenizer import SMILESTokenizer

	db_path = "data/zinc.lmdb"
	vocab_path = "config/vocab.yaml"
	max_len = 128

	# Build Tokenizer
	tokenizer = SMILESTokenizer.from_vocab_yaml(vocab_path)

	train_ds = ZincDataset(
		lmdb_path=db_path,
		subset_indices=ZincDataset.read_split_indices(db_path, "train"),
		tokenizer=tokenizer,
		max_length=max_len,
		is_training=True,
	)

	val_ds = ZincDataset(
		lmdb_path=db_path,
		subset_indices=ZincDataset.read_split_indices(db_path, "val"),
		tokenizer=tokenizer,
		max_length=max_len,
		is_training=False,
	)

	test_ds = ZincDataset(
		lmdb_path=db_path,
		subset_indices=ZincDataset.read_split_indices(db_path, "test"),
		tokenizer=tokenizer,
		max_length=max_len,
		is_training=False,
	)

	print(f"Splits created: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

	# Example of one item
	if len(train_ds) > 0:
		sample = train_ds[0]
		print("\nSample from training dataset:")
		for key, value in sample.items():
			print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

		# Decode for verification
		src_tokens = tokenizer.decode(sample["src"].numpy())
		tgt_tokens = tokenizer.decode(sample["tgt"].numpy())

		# Filter out special tokens for clean printing
		clean_src = "".join([t for t in src_tokens if t not in ["<PAD>", "<BOS>", "<EOS>"]])
		clean_tgt = "".join([t for t in tgt_tokens if t not in ["<PAD>", "<BOS>", "<EOS>"]])

		print(f"\nDecoded src (masked): {clean_src}")
		print(f"Decoded tgt (original): {clean_tgt}")
		print(f"Masked tokens count in src: {clean_src.count('<mask>')}")
		print(f"Span mask indicates masked: {torch.sum(sample['span_mask'])}")
