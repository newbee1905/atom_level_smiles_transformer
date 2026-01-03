import os
import sys
import time
import gzip
import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Generator, Dict, List, Any
from tqdm import tqdm
import httpx
import pandas as pd
from rdkit import Chem

from datasets import load_dataset
from chemformer_rs.tokenizer import RocksDBVocabBuilder

DATA_DIR = Path("./chemistry_datasets")
DB_DIR = Path("./rocksdb_databases")

DATA_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

MAX_WORKERS = max(1, os.cpu_count() - 1)

DATASETS = {
	"zinc": {
		"name": "ZINC 10M (Hugging Face)",
		"hf_repo": "jarod0411/zinc10M",
		"splits": ["train", "validation"],
		"column": "smiles",
		"description": "ZINC dataset (~10M molecules) via Hugging Face",
		"format": "HF_DATASET",
		"file": "zinc_hf_placeholder.txt",
	},
	"pubchem_subset": {
		"name": "PubChem Compound (Subset)",
		"url": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/Compound_099000001_099500000.sdf.gz",
		"file": "pubchem_subset.sdf.gz",
		"description": "PubChem compounds 99M-100M",
		"format": "SDF",
		"estimated_size_mb": 800,
	},
	"chembl": {
		"name": "ChEMBL 2025 (Hugging Face)",
		"hf_repo": "fabikru/chembl-2025-randomized-smiles-cleaned-rdkit-descriptors",
		"splits": ["train", "test"],
		"column": "smiles",
		"description": "ChEMBL 2025 Randomized SMILES via Hugging Face",
		"format": "HF_DATASET",
		"file": "chembl_hf_placeholder.txt",
	},
	"uspto": {
		"name": "USPTO Reactions (Local Pickle)",
		"url": None,
		"file": "data/uspto_sep.pickle",
		"description": "USPTO grant reactions from local pickle file (contains RDKit mol objects)",
		"format": "PICKLE_USPTO",
		"local_file": True,
		"estimated_size_mb": 2200,
	},
}


# -- Utils Functions --
def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
	"""Download file with progress bar using tqdm and httpx"""
	try:
		with httpx.stream("GET", url, timeout=30.0, follow_redirects=True) as response:
			response.raise_for_status()
			total_size = int(response.headers.get("content-length", 0))

			print(f"  Downloading: {url.split('/')[-1]}")

			with open(output_path, "wb") as f:
				with tqdm(
					total=total_size,
					unit="B",
					unit_scale=True,
					unit_divisor=1024,
					desc=f"  {output_path.name}",
					leave=True,
				) as pbar:
					for chunk in response.iter_bytes(chunk_size):
						if chunk:
							f.write(chunk)
							pbar.update(len(chunk))
			return True
	except Exception as e:
		print(f"  [Error] Download failed for {url}: {e}")
		return False


def decompress_gzip(gz_path: Path, output_path: Path) -> bool:
	"""Decompress gzip file"""
	try:
		print(f"  Decompressing: {gz_path.name}")
		total_size = gz_path.stat().st_size

		with gzip.open(gz_path, "rb") as f_in:
			with open(output_path, "wb") as f_out:
				# Using a larger buffer for speed
				while True:
					chunk = f_in.read(64 * 1024)
					if not chunk:
						break
					f_out.write(chunk)

		print(f"  [SUCCESS] Decompressed: {output_path.name}")
		return True
	except Exception as e:
		print(f"  [Error] Decompression failed: {e}")
		return False


# -- SMILES Extraction Functions --


def extract_smiles_from_hf_stream(dataset_config: dict, max_smiles: Optional[int] = None) -> Generator[str, None, None]:
	"""Streams SMILES from Hugging Face dataset."""
	repo = dataset_config["hf_repo"]
	splits = dataset_config["splits"]
	col_name = dataset_config["column"]
	total_yielded = 0

	for split in splits:
		try:
			ds = load_dataset(repo, split=split, streaming=True)

			if hasattr(ds, "select_columns"):
				try:
					ds = ds.select_columns([col_name])
				except Exception as e:
					print(f"  [Warning] Could not optimize column selection for {repo}: {e}")

			for sample in ds:
				if col_name in sample and sample[col_name]:
					smi = sample[col_name].strip()
					if len(smi) > 1:
						yield smi
						total_yielded += 1
						if max_smiles and total_yielded >= max_smiles:
							return
		except Exception as e:
			print(f"  [Error] reading HF split {split}: {e}")


def extract_smiles_from_csv(file_path: Path, max_smiles: Optional[int] = None) -> Generator[str, None, None]:
	"""Extract SMILES from CSV format"""
	try:
		line_count = 0
		# Increase field size limit for large CSVs (common in chemistry)
		csv.field_size_limit(sys.maxsize)

		with open(file_path, "r", encoding="utf-8") as f:
			reader = csv.reader(f)
			headers = next(reader, None)

			idx = 0
			if headers:
				lower_headers = [h.lower() for h in headers]
				if "smiles" in lower_headers:
					idx = lower_headers.index("smiles")
				elif "reactions" in lower_headers:
					idx = lower_headers.index("reactions")

			for row in reader:
				if row and len(row) > idx:
					smiles = row[idx].strip()
					# Basic validation filter
					if len(smiles) > 2 and any(c in smiles for c in "CNOSPFClBr"):
						yield smiles
						line_count += 1
						if max_smiles and line_count >= max_smiles:
							return
	except Exception as e:
		print(f"  [Error] reading CSV {file_path.name}: {e}")


def extract_smiles_from_sdf(file_path: Path, max_lines: Optional[int] = None) -> Generator[str, None, None]:
	"""Extract SMILES from SDF format"""
	try:
		count = 0
		with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
			while True:
				line = f.readline()
				if not line:
					break

				# Matches: > <canonical_smiles>, >  <SMILES>, etc.
				sline = line.strip()
				if sline.startswith(">") and "SMILES" in sline.upper():
					# The value is typically on the next line
					smi_line = f.readline()
					if not smi_line:
						break

					smi = smi_line.strip()

					# Basic validation: length > 1 and doesn't look like another tag or delimiter
					if len(smi) > 1 and not smi.startswith(">"):
						yield smi
						count += 1
						if max_lines and count >= max_lines:
							return

		if count == 0:
			print(f"  [Warning] No SMILES found in {file_path.name}. Check if tags match '> <...SMILES...>'")

	except Exception as e:
		print(f"  [Error] reading SDF {file_path.name}: {e}")


def mol_to_smiles_safe(mol: Optional[Any]) -> str:
	"""Safely convert a Mol object to canonical SMILES."""
	if not mol:
		return ""
	try:
		return Chem.MolToSmiles(mol, canonical=True)
	except Exception:
		return ""


def process_uspto_chunk(chunk: pd.DataFrame) -> List[str]:
	"""
	Worker function for parallel processing. Converts mol objects in a dataframe
	chunk to reaction SMILES strings.
	"""
	r_smi = [mol_to_smiles_safe(mol) for mol in chunk["reactants_mol"]]
	p_smi = [mol_to_smiles_safe(mol) for mol in chunk["products_mol"]]
	rg_smi = [mol_to_smiles_safe(mol) for mol in chunk["reagents_mol"]]

	results = []
	for i in range(len(r_smi)):
		if r_smi[i] and p_smi[i]:
			results.append(f"{r_smi[i]}>{rg_smi[i]}>{p_smi[i]}")
	return results


def extract_smiles_from_uspto_pickle(file_path: Path, max_smiles: Optional[int] = None) -> Generator[str, None, None]:
	"""
	Extracts reaction SMILES from the USPTO pickle file by converting mol objects
	in parallel and constructing the reaction string. This version uses a more
	memory-efficient parallelization strategy.
	"""
	print("  Loading USPTO pickle file (this may take a moment)...")
	try:
		df = pd.read_pickle(file_path)
		print(f"  Loaded dataframe with {len(df):,} reactions.")
	except Exception as e:
		print(f"  [Error] Could not read pickle file {file_path}: {e}")
		return

	chunk_size = 10000
	df_chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]

	print(f"  Starting SMILES extraction on {len(df_chunks)} chunks using {MAX_WORKERS} workers...")

	count = 0
	# Use a single Pool for all chunks, which is more memory-efficient
	with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
		for chunk in tqdm(df_chunks, desc="  Processing chunks"):
			try:
				r_smi = pool.map(mol_to_smiles_safe, chunk["reactants_mol"])
				p_smi = pool.map(mol_to_smiles_safe, chunk["products_mol"])
				rg_smi = pool.map(mol_to_smiles_safe, chunk["reagents_mol"])

				for i in range(len(r_smi)):
					if r_smi[i] and p_smi[i]:
						reaction_smiles = f"{r_smi[i]}>{rg_smi[i]}>{p_smi[i]}"
						yield reaction_smiles
						count += 1
						if max_smiles and count >= max_smiles:
							return
			except Exception as e:
				# This part of the code might not be reached if a worker segfaults,
				# but it's good practice to have it.
				print(f"  [Error] A chunk failed during processing: {e}")


# -- Dataset processing --
def process_dataset_task(args: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Worker function for ProcessPoolExecutor.
	Receives a dictionary of arguments to avoid pickling complex objects directly.
	"""

	dataset_name = args["name"]
	config = DATASETS[dataset_name]
	input_file = args["input_file"]
	output_smiles_file = args["output_file"]
	max_smiles = args.get("max_smiles")

	pid = multiprocessing.current_process().pid
	print(f"[PID {pid}] Starting extraction: {config['name']}")

	if config.get("format") == "HF_DATASET":
		extractor_func = extract_smiles_from_hf_stream
		input_source = config
	elif config["format"] == "PICKLE_USPTO":
		extractor_func = extract_smiles_from_uspto_pickle
		input_source = input_file
	elif config["format"] == "CSV":
		extractor_func = extract_smiles_from_csv
		input_source = input_file
	elif config["format"] == "SDF":
		extractor_func = extract_smiles_from_sdf
		input_source = input_file
	else:

		def simple_reader(p, m):
			with open(p, "r") as f:
				for l in f:
					yield l.split()[0]

		extractor_func = simple_reader
		input_source = input_file

	count = 0
	start_time = time.time()

	try:
		with open(output_smiles_file, "w") as out_f:
			for smiles in extractor_func(input_source, max_smiles):
				out_f.write(smiles + "\n")
				count += 1
				if count % 100000 == 0:
					pass

	except Exception as e:
		return {"status": "error", "dataset": dataset_name, "error": str(e)}

	elapsed = time.time() - start_time
	print(f"[PID {pid}] Finished {config['name']}: {count:,} SMILES in {elapsed:.1f}s")

	return {
		"status": "success",
		"dataset": dataset_name,
		"count": count,
		"elapsed": elapsed,
		"output_file": output_smiles_file,
	}


def download_dataset(dataset_name: str) -> Optional[Path]:
	"""
	Download/Prepare dataset. Handles local files that don't need downloading.
	"""

	if dataset_name not in DATASETS:
		return None

	dataset = DATASETS[dataset_name]
	print(f"\n[PREPARE] {dataset['name']}")

	if dataset.get("local_file", False):
		input_file = Path(dataset["file"])
		if input_file.exists():
			print(f"Found local file: {input_file}")
			return input_file
		else:
			print(f"  [Error] Local file not found: {input_file}")
			return None

	# For Hugging Face datasets, we just return a placeholder as they are streamed
	if dataset.get("format") == "HF_DATASET":
		return DATA_DIR / dataset["file"]

	# For other downloadable files
	input_file = DATA_DIR / dataset["file"]
	output_name = dataset["file"].replace(".tar.gz", "_extracted").replace(".gz", "")
	output_path = DATA_DIR / output_name

	if output_path.exists():
		if output_path.is_dir():
			return output_path
		if output_path.stat().st_size > 0:
			return output_path

	if not input_file.exists():
		if not download_file(dataset["url"], input_file):
			return None

	# Decompress if needed
	is_gzip = dataset["file"].endswith(".gz") and not dataset["file"].endswith(".tar.gz")

	if is_gzip:
		if not decompress_gzip(input_file, output_path):
			return None
	else:
		output_path = input_file

	return output_path


# -- Vocab building --
def build_vocab_task(args: Dict[str, Any]) -> Dict[str, Any]:
	"""Worker function for building individual RocksDB vocabs"""
	dataset_name = args["name"]
	smiles_file = args["smiles_file"]
	chunk_size = args.get("chunk_size", 100000)

	pid = multiprocessing.current_process().pid
	print(f"[PID {pid}] Building Vocab: {dataset_name}")

	db_path = DB_DIR / f"vocab_{dataset_name}"

	try:
		# Import here to ensure class is available in worker
		from chemformer_rs.tokenizer import RocksDBVocabBuilder

		builder = RocksDBVocabBuilder(str(db_path), chunk_size=chunk_size)
		start_time = time.time()

		vocab, _, (total_tokens, unique, total_smiles) = builder.build_from_file(
			str(smiles_file), num_threads=4, export_json=str(db_path / "vocab_metadata.json")
		)

		elapsed = time.time() - start_time
		print(f"[PID {pid}] Vocab Built {dataset_name}: {total_tokens:,} tokens ({elapsed:.1f}s)")

		return {"status": "success", "dataset": dataset_name, "unique_tokens": unique}
	except Exception as e:
		return {"status": "error", "dataset": dataset_name, "error": str(e)}


def merge_smiles_files(file_paths: list, output_file: Path):
	"""Merge files (Sequential is fine for IO bound concatenation)"""
	print(f"\n[MERGE] Merging {len(file_paths)} files into {output_file.name}")
	with open(output_file, "w") as out_f:
		for fp in file_paths:
			with open(fp, "r") as in_f:
				for line in in_f:
					out_f.write(line)


def build_unified_vocab(chunk_size: int = 500000):
	"""Build unified vocabulary from all datasets"""
	print(f"\n[BUILD UNIFIED VOCAB]")

	smiles_files = [
		DATA_DIR / f"{name}_smiles.txt" for name in DATASETS.keys() if (DATA_DIR / f"{name}_smiles.txt").exists()
	]

	if not smiles_files:
		print("Error: No SMILES files found.")
		return

	unified_file = DATA_DIR / "all_datasets_merged.smi"
	merge_smiles_files(smiles_files, unified_file)

	db_path = DB_DIR / "vocab_unified_all_datasets"

	try:
		from chemformer_rs.tokenizer import RocksDBVocabBuilder

		builder = RocksDBVocabBuilder(str(db_path), chunk_size=chunk_size)

		print("  Starting unified build...")
		builder.build_from_file(str(unified_file), num_threads=8, export_json=str(db_path / "vocab_metadata.json"))
		print("[SUCCESS] Unified vocabulary built.")
	except Exception as e:
		print(f"Error building unified vocab: {e}")


def main():
	print("=" * 80)
	print(f"CHEMISTRY VOCAB BUILDER | Parallel Workers: {MAX_WORKERS}")
	print("=" * 80)

	print("\n>>> STEP 1: Preparing Datasets")
	ready_datasets = {}  # name: input_file_path

	for name in DATASETS.keys():
		path = download_dataset(name)
		if path:
			ready_datasets[name] = path

	print("\n>>> STEP 2: Extracting SMILES (Parallel)")
	extraction_tasks = []

	# Prepare arguments
	for name, input_path in ready_datasets.items():
		output_file = DATA_DIR / f"{name}_smiles.txt"
		extraction_tasks.append(
			{"name": name, "input_file": input_path, "output_file": output_file, "max_smiles": None}
		)

	smiles_files_map = {}

	with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
		futures = {executor.submit(process_dataset_task, task): task["name"] for task in extraction_tasks}

		for future in as_completed(futures):
			name = futures[future]
			try:
				result = future.result()
				if result["status"] == "success":
					smiles_files_map[name] = result["output_file"]
				else:
					print(f"[ERROR] Task failed for {name}: {result.get('error')}")
			except Exception as e:
				print(f"[ERROR] Exception in task {name}: {e}")

	print("\n>>> STEP 3: Building Individual Vocabularies (Parallel)")
	vocab_tasks = []
	for name, s_file in smiles_files_map.items():
		vocab_tasks.append({"name": name, "smiles_file": s_file, "chunk_size": 50000})

	with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
		futures = {executor.submit(build_vocab_task, task): task["name"] for task in vocab_tasks}
		for future in as_completed(futures):
			future.result()

	print("\n>>> STEP 4: Unified Vocabulary")
	build_unified_vocab()

	print("\n" + "=" * 80)
	print("ALL TASKS COMPLETED")


if __name__ == "__main__":
	multiprocessing.freeze_support()  # For Windows support
	main()
