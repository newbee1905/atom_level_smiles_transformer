import os
import pickle
import numpy as np
import torch
from rdkit import Chem

from dataset.utils import LMDBDataset, prepare_sequence

class UsptoSepDataset(LMDBDataset):
	def __init__(
		self,
		lmdb_path,
		tokenizer,
		max_length,
		subset_indices=None,
		augment_prob=1.0,
		is_training=False,
		retro=False,
	):
		super().__init__(lmdb_path, max_length, subset_indices, is_training)
		self.tokenizer = tokenizer
		self.augment_prob = augment_prob
		self.retro = retro

		self.pad_token_id = self.tokenizer.token_to_index("<PAD>")
		self.bos_token_id = self.tokenizer.token_to_index("<BOS>")
		self.eos_token_id = self.tokenizer.token_to_index("<EOS>")
		self.unk_token_id = self.tokenizer.token_to_index("<UNK>")

	def _mol_to_smiles(self, mol):
		if not mol:
			return ""
		is_random = self.is_training and np.random.rand() < self.augment_prob
		if is_random:
			return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
		else:
			return Chem.MolToSmiles(mol, doRandom=False, canonical=True)

	def __getitem__(self, index):
		if self.env is None:
			self._init_db()

		db_index = self.indices[index]
		with self.env.begin(write=False) as txn:
			pickled_data = txn.get(f"{db_index}".encode("ascii"))

		data = pickle.loads(pickled_data)
		reactants_mol = data["reactants_mol"]
		reagents_mol = data["reagents_mol"]
		products_mol = data["products_mol"]

		reactants_smi_mapped = self._mol_to_smiles(reactants_mol)
		reagents_smi_mapped = self._mol_to_smiles(reagents_mol) if reagents_mol else ""
		products_smi_mapped = self._mol_to_smiles(products_mol)

		if reagents_smi_mapped:
			fwd_source_smi = f"{reactants_smi_mapped}>{reagents_smi_mapped}"
			retro_target_smi = f"{reactants_smi_mapped}>{reagents_smi_mapped}"
		else:
			fwd_source_smi = reactants_smi_mapped
			retro_target_smi = reactants_smi_mapped

		fwd_target_smi = products_smi_mapped
		retro_source_smi = products_smi_mapped

		source_smiles = retro_source_smi if self.retro else fwd_source_smi
		target_smiles = retro_target_smi if self.retro else fwd_target_smi

		src_core_ids, _ = self.tokenizer.encode(
			source_smiles, add_bos=False, add_eos=False, pad_to_length=False, max_length=self.max_length - 2
		)
		tgt_core_ids, _ = self.tokenizer.encode(
			target_smiles, add_bos=False, add_eos=False, pad_to_length=False, max_length=self.max_length - 2
		)
		src_core_ids = np.array(src_core_ids)
		tgt_core_ids = np.array(tgt_core_ids)

		src_ids, src_attention_mask = prepare_sequence(
			src_core_ids, self.max_length, self.bos_token_id, self.eos_token_id, self.pad_token_id
		)
		tgt_ids, tgt_attention_mask = prepare_sequence(
			tgt_core_ids, self.max_length, self.bos_token_id, self.eos_token_id, self.pad_token_id
		)

		return {
			"src": torch.from_numpy(src_ids),
			"src_attention_mask": torch.from_numpy(src_attention_mask),
			"tgt": torch.from_numpy(tgt_ids),
			"tgt_attention_mask": torch.from_numpy(tgt_attention_mask),
		}


if __name__ == "__main__":
	from chemformer_rs.tokenizer import SMILESTokenizer

	db_path = "data/uspto_sep.lmdb"
	vocab_path = "config/vocab.yaml"
	max_len = 256

	print("Checking if LMDB exists...")
	if not os.path.exists(db_path):
		print(f"LMDB not found at {db_path}.")
		print("Please run scripts/uspto_pickle_to_lmdb.py first to generate it.")
	else:
		tokenizer = SMILESTokenizer.from_vocab_yaml(vocab_path)
		if ">" not in tokenizer.tokens:
			tokenizer.add_token(">")

		# --- Test Forward mode ---
		print("\n--- Testing Forward Mode ---")
		fwd_ds = UsptoSepDataset(
			lmdb_path=db_path,
			subset_indices=UsptoSepDataset.read_split_indices(db_path, "val")[:1],
			tokenizer=tokenizer,
			max_length=max_len,
			is_training=True,
			retro=False,
		)
		if len(fwd_ds) > 0:
			sample = fwd_ds[0]
			print("Sample (forward):")
			for key, value in sample.items():
				print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
			src_toks = "".join(
				[t for t in tokenizer.decode(sample["src"].numpy()) if t not in ["<PAD>", "<BOS>", "<EOS>"]]
			)
			tgt_toks = "".join(
				[t for t in tokenizer.decode(sample["tgt"].numpy()) if t not in ["<PAD>", "<BOS>", "<EOS>"]]
			)
			print(f"  Decoded src: {src_toks}")
			print(f"  Decoded tgt: {tgt_toks}")

		# --- Test Retro mode ---
		print("\n--- Testing Retro mode ---")
		retro_ds = UsptoSepDataset(
			lmdb_path=db_path,
			subset_indices=UsptoSepDataset.read_split_indices(db_path, "val")[:1],
			tokenizer=tokenizer,
			max_length=max_len,
			is_training=True,
			retro=True,
		)
		if len(retro_ds) > 0:
			sample = retro_ds[0]
			print("Sample (retro):")
			for key, value in sample.items():
				print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
			src_toks = "".join(
				[t for t in tokenizer.decode(sample["src"].numpy()) if t not in ["<PAD>", "<BOS>", "<EOS>"]]
			)
			tgt_toks = "".join(
				[t for t in tokenizer.decode(sample["tgt"].numpy()) if t not in ["<PAD>", "cb", "<BOS>", "<EOS>"]]
			)
			print(f"  Decoded src: {src_toks}")
			print(f"  Decoded tgt: {tgt_toks}")
