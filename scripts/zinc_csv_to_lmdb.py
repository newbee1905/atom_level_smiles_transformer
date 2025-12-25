import os
import glob
import lmdb
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import io

import zstandard as zstd
from pyroaring import BitMap


def count_rows_fast(csv_path):
	"""
	Counts rows using a fast binary read to estimate total for tqdm.
	Subtracts 1 for the header.

	References:
	- Fast line counting in Python: https://stackoverflow.com/q/19001402
	"""
	count = 0
	try:
		with open(csv_path, "rb") as f:
			buf_size = 1024 * 1024
			read_f = f.raw.read
			buf = read_f(buf_size)
			while buf:
				count += buf.count(b"\n")
				buf = read_f(buf_size)
		return max(0, count - 1)
	except Exception:
		return 0


def main():
	parser = argparse.ArgumentParser(description="Single-threaded ZINC CSV to LMDB converter (Baseline)")
	parser.add_argument("--input", required=True, help="Directory containing .csv files")
	parser.add_argument("--output", default="zinc_single.lmdb", help="Path to output LMDB file")
	parser.add_argument("--map_size", type=int, default=10**12, help="LMDB map size (default 1TB)")
	parser.add_argument("--chunk_size", type=int, default=50000, help="Rows per batch commit")
	args = parser.parse_args()

	csv_files = glob.glob(os.path.join(args.input, "*.csv"))
	if not csv_files:
		print("No CSV files found in the specified directory.")
		return

	print(f"Scanning {len(csv_files)} files...")
	total_expected = 0
	for f in tqdm(csv_files, desc="Indexing"):
		total_expected += count_rows_fast(f)
	print(f"Estimated total molecules: {total_expected:,}")

	env = lmdb.open(
		args.output,
		map_size=args.map_size,
		writemap=True,
		map_async=True,
		meminit=False,
	)

	count = 0
	txn = env.begin(write=True)

	# Dictionary to track indices for each split (e.g., 'train', 'val', 'test')
	split_indices = defaultdict(BitMap)

	try:
		with tqdm(
			total=total_expected,
			desc="Converting (Single-threaded)",
			dynamic_ncols=True,
		) as pbar:
			for file_path in csv_files:
				try:
					# Read in chunks to manage memory
					reader = pd.read_csv(file_path, usecols=["smiles", "set"], chunksize=args.chunk_size)

					for df in reader:
						df.columns = [c.strip().lower() for c in df.columns]

						smiles_list = df["smiles"].tolist()
						sets_list = df["set"].tolist()

						for i in range(len(smiles_list)):
							split_name = str(sets_list[i])
							split_indices[split_name].add(count)

							key = f"{count}".encode("ascii")
							value = str(smiles_list[i]).encode("utf-8")
							txn.put(key, value)

							count += 1

							# Commit periodically
							if count % args.chunk_size == 0:
								txn.commit()
								txn = env.begin(write=True)

						pbar.update(len(df))

				except Exception as e:
					print(f"\n[Error] Failed to process {file_path}: {e}")

		cctx = zstd.ZstdCompressor()
		for split_name, bitmap in split_indices.items():
			split_key = f"split_{split_name}".encode("ascii")

			# Serialize the bitmap to bytes
			raw_bytes = bitmap.serialize()

			compressed_indices = cctx.compress(raw_bytes)
			txn.put(split_key, compressed_indices)
			print(
				f" - Storing compressed '{split_name}' indices ({len(raw_bytes):,} bytes -> {len(compressed_indices):,} bytes)"
			)

		txn.put(b"__len__", str(count).encode("ascii"))
		txn.commit()
		env.sync()
		env.close()

		print(f"\nSuccessfully converted data to {args.output}")
		print(f"Total records written: {count:,}")
		for split_name, indices in split_indices.items():
			print(f" - {split_name}: {len(indices):,} items")

	except KeyboardInterrupt:
		print("\nProcess interrupted by user.")
		txn.abort()
		env.close()


if __name__ == "__main__":
	main()
