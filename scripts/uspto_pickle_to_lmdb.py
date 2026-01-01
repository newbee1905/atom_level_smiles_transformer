import os
import lmdb
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pickle
from rdkit import Chem

import zstandard as zstd
from pyroaring import BitMap


from multiprocessing import Pool, cpu_count
import numpy as np


def mol_to_smiles_safe(mol):
	"""Safely convert a Mol object to canonical SMILES."""
	if not mol:
		return ""
	try:
		return Chem.MolToSmiles(mol, canonical=True)
	except Exception:
		return ""


def main():
	parser = argparse.ArgumentParser(description="Optimized and parallel USPTO Pickle to LMDB converter.")
	parser.add_argument("--input", required=True, help="Path to input .pkl file")
	parser.add_argument("--output", default="data/uspto.lmdb", help="Path to output LMDB file")
	parser.add_argument("--map_size", type=int, default=10**12, help="LMDB map size (default 1TB)")
	parser.add_argument(
		"--chunk_size",
		type=int,
		default=10000,
		help="Rows per batch commit to LMDB",
	)
	args = parser.parse_args()

	if not os.path.exists(args.input):
		print(f"Input file not found: {args.input}")
		return

	print("Loading pickle file...")
	df = pd.read_pickle(args.input)
	print(f"Loaded dataframe with {len(df):,} reactions.")

	# --- Process in Chunks to conserve memory ---
	num_procs = max(1, cpu_count() - 2)
	# More, smaller chunks are better for memory. Let's aim for chunks of ~10k rows.
	num_chunks = len(df) // 10000
	if num_chunks == 0:
		num_chunks = 1
	df_chunks = np.array_split(df, num_chunks)

	print(f"Starting SMILES conversion with {num_procs} processes, processing {len(df_chunks)} chunks...")

	processed_chunks = []
	with Pool(num_procs) as pool:
		for chunk in tqdm(df_chunks, desc="Processing Chunks"):
			# Use pool.map which is simpler for this chunk-based approach
			r_smi = pool.map(mol_to_smiles_safe, chunk["reactants_mol"])
			p_smi = pool.map(mol_to_smiles_safe, chunk["products_mol"])
			rg_smi = pool.map(mol_to_smiles_safe, chunk["reagents_mol"])

			chunk["reactants_smi"] = r_smi
			chunk["products_smi"] = p_smi
			chunk["reagents_smi"] = rg_smi
			processed_chunks.append(chunk)

	print("Finished SMILES conversion. Concatenating chunks...")
	df = pd.concat(processed_chunks, ignore_index=True)
	del processed_chunks  # Free memory

	# --- Filtering and Deduplication (now on the full DataFrame with SMILES) ---
	print("Filtering invalid and duplicate reactions...")
	original_count = len(df)
	df = df[(df["reactants_smi"] != "") & (df["products_smi"] != "")].copy()
	invalid_count = original_count - len(df)
	print(f"  - Filtered out {invalid_count:,} rows with invalid reactants or products.")

	df["reaction_signature"] = df["reactants_smi"] + ">" + df["reagents_smi"] + ">" + df["products_smi"]
	original_count = len(df)
	df.drop_duplicates(subset=["reaction_signature"], keep="first", inplace=True)
	duplicate_count = original_count - len(df)
	print(f"  - Filtered out {duplicate_count:,} duplicate reactions.")

	df.reset_index(drop=True, inplace=True)
	final_count = len(df)
	print(f"A total of {final_count:,} unique, valid reactions will be written to LMDB.")

	# --- Write to LMDB ---
	env = lmdb.open(
		args.output,
		map_size=args.map_size,
		writemap=True,
		map_async=True,
		meminit=False,
	)
	txn = env.begin(write=True)
	split_indices = defaultdict(BitMap)

	records_to_write = df[["reactants_mol", "products_mol", "reagents_mol", "set"]].to_dict("records")

	for idx, record in tqdm(enumerate(records_to_write), total=final_count, desc="Writing to LMDB"):
		split_name = str(record["set"])
		# Normalize split names
		if split_name in ["valid", "validation"]:
			split_name = "val"
		split_indices[split_name].add(idx)

		key = f"{idx}".encode("ascii")
		data_to_store = {
			"reactants_mol": record["reactants_mol"],
			"products_mol": record["products_mol"],
			"reagents_mol": record["reagents_mol"],
		}
		value = pickle.dumps(data_to_store)
		txn.put(key, value)

		if (idx + 1) % args.chunk_size == 0:
			txn.commit()
			txn = env.begin(write=True)

	# --- Finalize DB ---
	print("Finalizing database...")
	txn.put(b"__len__", str(final_count).encode("ascii"))

	cctx = zstd.ZstdCompressor()
	for split_name, bitmap in split_indices.items():
		split_key = f"split_{split_name}".encode("ascii")
		raw_bytes = bitmap.serialize()
		compressed_indices = cctx.compress(raw_bytes)
		txn.put(split_key, compressed_indices)
		print(
			f"  - Storing compressed '{split_name}' indices ({len(raw_bytes):,} bytes -> {len(compressed_indices):,} bytes)"
		)

	txn.commit()
	env.sync()
	env.close()

	print(f"\nSuccessfully converted data to {args.output}")
	print(f"Total records written: {final_count:,}")
	for split_name, indices in split_indices.items():
		print(f"  - {split_name}: {len(indices):,} items")


if __name__ == "__main__":
	main()
