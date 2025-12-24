import argparse
import lmdb
import zstandard as zstd
import pickle
import os
from tqdm import tqdm

def get_entry_count(env: lmdb.Environment) -> int:
	"""Get the total number of entries in an LMDB database."""
	with env.begin() as txn:
		return txn.stat()["entries"]


def compress_lmdb(args: argparse.Namespace):
	"""Compresses an LMDB database to an .lmdb.zst file."""
	if not os.path.isdir(args.input):
		print(f"Error: Input LMDB path '{args.input}' not found or is not a directory.")
		return

	print(f"Opening LMDB database at '{args.input}'...")
	try:
		env = lmdb.open(
			args.input,
			readonly=True,
			lock=False,
			readahead=False,
			meminit=False,
		)
	except lmdb.Error as e:
		print(f"Error opening LMDB database: {e}")
		return

	num_entries = get_entry_count(env)
	print(f"Found {num_entries:,} entries to compress.")

	cctx = zstd.ZstdCompressor(level=args.level)
	print(f"Compressing to '{args.output}' using zstd (level={args.level})...")
	try:
		with open(args.output, "wb") as f_out, cctx.stream_writer(f_out) as compressor:
			# Write the number of entries as the first item for the decompressor's progress bar.
			pickle.dump(num_entries, compressor, protocol=pickle.HIGHEST_PROTOCOL)

			with env.begin() as txn:
				cursor = txn.cursor()
				with tqdm(cursor, total=num_entries, desc="Compressing", unit="recs") as pbar:
					for key, value in pbar:
						pickle.dump((key, value), compressor, protocol=pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print(f"An error occurred during compression: {e}")
	finally:
		env.close()

	print("Compression complete.")


def decompress_lmdb(args: argparse.Namespace):
	"""Decompresses an .lmdb.zst file to an LMDB database."""
	if not os.path.isfile(args.input):
		print(f"Error: Input file '{args.input}' not found.")
		return

	if os.path.exists(args.output):
		print(f"Error: Output path '{args.output}' already exists. Please remove it first.")
		return

	print(f"Decompressing '{args.input}' to LMDB at '{args.output}'...")
	try:
		env = lmdb.open(
			args.output,
			map_size=args.map_size,
			writemap=True,
			map_async=True,
			meminit=False,
		)
	except lmdb.Error as e:
		print(f"Error creating new LMDB database: {e}")
		return

	try:
		with open(args.input, "rb") as f_in, zstd.ZstdDecompressor().stream_reader(f_in) as reader:
			unpickler = pickle.Unpickler(reader)
			
			# Read the number of entries for the progress bar.
			num_entries = unpickler.load()

			with tqdm(total=num_entries, desc="Decompressing", unit="recs") as pbar:
				txn = env.begin(write=True)
				commit_count = 0
				
				for i in range(num_entries):
					key, value = unpickler.load()
					txn.put(key, value)
					commit_count += 1
					pbar.update(1)

					if commit_count % args.chunk_size == 0:
						txn.commit()
						pbar.set_description(f"Decompressing (commit {i // args.chunk_size})")
						txn = env.begin(write=True)
				
				txn.commit()
				env.sync()

	except (pickle.UnpicklingError, zstd.ZstdError) as e:
		print(f"An error occurred during decompression. The file may be corrupt. Error: {e}")
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
	finally:
		env.close()

	print("Decompression complete.")


def main():
	parser = argparse.ArgumentParser(
		description="Compress and decompress LMDB databases for easy transfer using Zstandard.",
		formatter_class=argparse.RawTextHelpFormatter,
	)
	subparsers = parser.add_subparsers(dest="command", required=True)

	# Compression 
	parser_compress = subparsers.add_parser("compress", help="Compress an LMDB database folder into a single .zst file.")
	parser_compress.add_argument("--input", required=True, help="Path to the input LMDB database directory.")
	parser_compress.add_argument("--output", required=True, help="Path for the output compressed file (e.g., data.lmdb.zst).")
	parser_compress.add_argument(
		"--level", type=int, default=3, help="Zstandard compression level. 1 is fastest, 22 is highest compression. (default: 3)"
	)
	parser_compress.set_defaults(func=compress_lmdb)

	# Decompression 
	parser_decompress = subparsers.add_parser("decompress", help="Decompress a .zst file into an LMDB database folder.")
	parser_decompress.add_argument("--input", required=True, help="Path to the input compressed .zst file.")
	parser_decompress.add_argument("--output", required=True, help="Path for the output LMDB database directory.")
	parser_decompress.add_argument(
		"--map_size", type=int, default=10**12, help="LMDB map size for the new database (default 1TB)."
	)
	parser_decompress.add_argument(
		"--chunk_size", type=int, default=50000, help="Records per LMDB commit during decompression."
	)
	parser_decompress.set_defaults(func=decompress_lmdb)

	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
