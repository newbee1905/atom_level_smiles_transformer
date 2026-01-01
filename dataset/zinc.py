import torch
import numpy as np
from rdkit import Chem

from dataset.utils import LMDBDataset, prepare_sequence

class ZincDataset(LMDBDataset):
	def __init__(
		self,
		lmdb_path,
		tokenizer,
		max_length,
		subset_indices=None,
		mask_prob=0.30,
		span_len=3,
		augment_prob=1.0,
		is_training=False,
		span_mask_proportion=1.0,
		span_random_proportion=0.0,
	):
		super().__init__(lmdb_path, max_length, subset_indices, is_training)
		self.tokenizer = tokenizer
		self.mask_prob = mask_prob
		self.span_len = span_len
		self.augment_prob = augment_prob
		self.span_mask_proportion = span_mask_proportion
		self.span_random_proportion = span_random_proportion

		# Get special token IDs
		self.mask_token_id = self.tokenizer.token_to_index("<MASK>")
		self.pad_token_id = self.tokenizer.token_to_index("<PAD>")
		self.bos_token_id = self.tokenizer.token_to_index("<BOS>")
		self.eos_token_id = self.tokenizer.token_to_index("<EOS>")
		self.unk_token_id = self.tokenizer.token_to_index("<UNK>")
		self.vocab_size = self.tokenizer.vocab_size
		self.special_token_ids = {
			self.mask_token_id,
			self.pad_token_id,
			self.bos_token_id,
			self.eos_token_id,
			self.unk_token_id,
		}

	def randomize_smiles(self, smiles):
		"""Default RDKit randomization."""
		mol = Chem.MolFromSmiles(smiles)
		return Chem.MolToSmiles(mol, doRandom=True, canonical=False) if mol else smiles

	def apply_span_masking(self, token_ids: np.ndarray):
		"""
		Applies a mixed-strategy span masking to the token sequence.
		Spans of tokens are replaced by a single token, which can be either a
		special <MASK> token or a random token from the vocabulary, based on
		the configured proportions.

		Returns:
			- new_tokens (np.ndarray): The corrupted token sequence.
			- to_mask (np.ndarray): A boolean mask aligned with the *original*
			  token sequence, indicating which tokens were part of a masked span.
			- is_fake (np.ndarray): A boolean mask aligned with the *new*
			  (corrupted) token sequence, indicating which tokens are "fake".
		"""
		n = len(token_ids)

		if n == 0:
			return token_ids, np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

		# Determine which tokens are part of a mask span
		to_mask = np.zeros(n, dtype=bool)
		num_masked = 0
		num_to_mask = int(round(n * self.mask_prob))

		if num_to_mask == 0:
			return token_ids, to_mask, np.zeros(n, dtype=bool)

		while num_masked < num_to_mask:
			# Clamp span_length so it cannot exceed n
			sampled_len = max(1, np.random.poisson(self.span_len))
			span_length = min(sampled_len, n)

			start = np.random.randint(0, n - span_length + 1)

			to_mask[start : start + span_length] = True
			num_masked = np.sum(to_mask)

		# Collapse spans into single tokens and generate fake mask
		new_tokens = []
		is_fake = []

		i = 0
		while i < n:
			if to_mask[i]:
				# Start of a masked span, this new token is "fake"
				is_fake.append(True)

				# Decide whether to use <MASK> or a random token
				if np.random.rand() < self.span_mask_proportion:
					new_tokens.append(self.mask_token_id)
				else:
					# Sample a random token, avoiding special tokens
					while True:
						random_token_id = np.random.randint(0, self.vocab_size)
						if random_token_id not in self.special_token_ids:
							break
					new_tokens.append(random_token_id)

				# Skip all contiguous tokens in this masked span
				while i < n and to_mask[i]:
					i += 1
			else:
				# Not masked, keep original and mark as not "fake"
				new_tokens.append(token_ids[i])
				is_fake.append(False)
				i += 1

		return np.array(new_tokens), to_mask, np.array(is_fake)

	def __getitem__(self, index):
		if self.env is None:
			self._init_db()

		db_index = self.indices[index]
		with self.env.begin(write=False) as txn:
			smi = txn.get(f"{db_index}".encode("ascii")).decode("ascii")

		augmented_smiles = (
			(self.randomize_smiles(smi) if np.random.rand() < self.augment_prob else smi) if self.is_training else smi
		)

		core_token_ids, _ = self.tokenizer.encode(
			augmented_smiles,
			add_bos=False,
			add_eos=False,
			pad_to_length=False,
			max_length=self.max_length - 2,  # Leave space for BOS/EOS
		)
		core_token_ids = np.array(core_token_ids)

		# Apply span masking to create the source sequence
		masked_core_token_ids, span_mask_unpadded, is_fake_unpadded = self.apply_span_masking(core_token_ids)

		src_ids, src_attention_mask = prepare_sequence(
			masked_core_token_ids, self.max_length, self.bos_token_id, self.eos_token_id, self.pad_token_id
		)
		tgt_ids, tgt_attention_mask = prepare_sequence(
			core_token_ids, self.max_length, self.bos_token_id, self.eos_token_id, self.pad_token_id
		)

		# Prepare electra_labels (aligned with src)
		seq_len_src = len(masked_core_token_ids) + 2
		padding_len_src = self.max_length - seq_len_src
		electra_labels = np.pad(
			is_fake_unpadded,
			(1, 1 + padding_len_src),  # Pad for BOS, EOS, and padding
			"constant",
			constant_values=False,
		)

		# Pad the span mask (aligned with tgt), adding False for special tokens and padding
		seq_len_tgt = len(core_token_ids) + 2
		padding_len_tgt = self.max_length - seq_len_tgt
		span_mask = np.pad(
			span_mask_unpadded,
			(1, 1 + padding_len_tgt),
			"constant",
			constant_values=False,
		)

		return {
			"src": torch.from_numpy(src_ids),
			"src_attention_mask": torch.from_numpy(src_attention_mask),
			"tgt": torch.from_numpy(tgt_ids),
			"tgt_attention_mask": torch.from_numpy(tgt_attention_mask),
			"span_mask": torch.from_numpy(span_mask),
			"electra_labels": torch.from_numpy(electra_labels).float(),
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
