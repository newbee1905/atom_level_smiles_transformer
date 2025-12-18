import torch
import numpy as np
import lmdb
import pickle
from torch.utils.data import Dataset
from rdkit import Chem


class ZincDataset(Dataset):
	def __init__(
		self,
		lmdb_path,
		subset_indices=None,
		mask_prob=0.15,
		span_len=3,
		augment_prob=0.5,
		is_training=False,
	):
		self.lmdb_path = lmdb_path
		self.mask_prob = mask_prob
		self.span_len = span_len
		self.augment_prob = augment_prob
		self.is_training = is_training

		self.indices = subset_indices

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

	def apply_span_masking(self, smiles):
		"""Replaces contiguous spans of characters with <mask>."""
		tokens = list(smiles)
		n = len(tokens)
		num_to_mask = max(1, int(n * self.mask_prob))
		masked_indices = set()

		while len(masked_indices) < num_to_mask:
			start = np.random.randint(0, n)
			length = np.random.randint(1, self.span_len + 1)
			for i in range(start, min(start + length, n)):
				masked_indices.add(i)
				if len(masked_indices) >= num_to_mask:
					break

		return "".join([tokens[i] if i not in masked_indices else "<mask>" for i in range(n)])

	def __getitem__(self, index):
		with self.env.begin(write=False) as txn:
			data = pickle.loads(txn.get(f"{index}".encode("ascii")))

		original = data["smiles"]

		augmented = self.randomize_smiles(original) if np.random.rand() < self.augment_prob else original
		masked = self.apply_span_masking(augmented)

		return {
			"zinc_id": data["zinc_id"],
			"original": original,
			"masked": masked,
			"set": data["set"],
		}


if __name__ == "__main__":
	db_path = "data/zinc.lmdb"
	train_ds = ZincDataset(
		db_path,
		subset_indices=ZincDataset.read_split_indices(db_path, "train"),
		is_training=True,
		augment_prob=0.5,
	)

	val_ds = ZincDataset(
		db_path,
		subset_indices=ZincDataset.read_split_indices(db_path, "val"),
		is_training=False,
		augment_prob=0.0,
	)

	test_ds = ZincDataset(
		db_path,
		subset_indices=ZincDataset.read_split_indices(db_path, "test"),
		is_training=False,
		augment_prob=0.0,
	)

	print(f"Splits created: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
