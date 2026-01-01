import lmdb
import numpy as np
import zstandard as zstd
from pyroaring import BitMap
from torch.utils.data import Dataset
import torch


def prepare_sequence(token_ids, max_length, bos_token_id, eos_token_id, pad_token_id):
	"""
	Adds BOS/EOS tokens, pads to max_length, and creates an attention mask.
	Also handles truncation if the sequence is too long.
	"""
	token_ids = np.concatenate([[bos_token_id], token_ids, [eos_token_id]]).astype(int)
	seq_len = len(token_ids)
	padding_len = max_length - seq_len

	if padding_len < 0:
		token_ids = token_ids[: max_length - 1]
		token_ids[-1] = eos_token_id
		seq_len = max_length
		padding_len = 0

	padded_ids = np.pad(token_ids, (0, padding_len), "constant", constant_values=pad_token_id)
	attention_mask = np.pad(np.ones(seq_len, dtype=int), (0, padding_len), "constant", constant_values=0)

	return padded_ids, attention_mask


class LMDBDataset(Dataset):
	def __init__(self, lmdb_path, max_length, subset_indices=None, is_training=False):
		self.lmdb_path = lmdb_path
		self.max_length = max_length
		self.is_training = is_training
		self.env = None

		env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
		with env.begin(write=False) as txn:
			self.full_length = int(txn.get(b"__len__").decode("ascii"))
		env.close()

		if subset_indices is None:
			self.indices = list(range(self.full_length))
		else:
			self.indices = subset_indices

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

	@staticmethod
	def read_split_indices(lmdb_path, split_name):
		env = lmdb.open(lmdb_path, readonly=True, lock=False)
		with env.begin(write=False) as txn:
			compressed_indices = txn.get(f"split_{split_name}".encode("ascii"))
			if compressed_indices is None:
				raise ValueError(f"Split '{split_name}' not found in LMDB at {lmdb_path}")

			raw_bytes = zstd.decompress(compressed_indices)
			bitmap = BitMap.deserialize(raw_bytes)
			split_indices = np.array(bitmap, dtype=np.uint32)
		env.close()
		return split_indices
