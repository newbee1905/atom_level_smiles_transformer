import pytest
import shutil
import tempfile
import lmdb
import os
import pickle
import zlib
from chemformer_rs.database import PyRockDB

# --- Utility Functions ---
def get_dir_size_rocksdb(path):
	"""Standard size calculation for RocksDB (sum of all files)."""
	total = 0
	if os.path.isfile(path):
		return os.path.getsize(path)
	for dirpath, _, filenames in os.walk(path):
		for f in filenames:
			fp = os.path.join(dirpath, f)
			if os.path.exists(fp):
				total += os.path.getsize(fp)
	return total


def get_actual_lmdb_size(path):
	"""Calculates physical disk usage of LMDB directory using block counts (sparse-aware)."""
	total_bytes = 0
	for filename in ["data.mdb", "lock.mdb"]:
		fp = os.path.join(path, filename)
		if os.path.exists(fp):
			stat = os.stat(fp)
			# Using st_blocks * 512 provides an accurate measure of sparse file disk usage
			total_bytes += stat.st_blocks * 512
	return total_bytes


def format_size(bytes_size):
	"""Format bytes to human readable format."""
	for unit in ["B", "KB", "MB", "GB"]:
		if bytes_size < 1024.0:
			return f"{bytes_size:.2f} {unit}"
		bytes_size /= 1024.0
	return f"{bytes_size:.2f} TB"


# --- Fixtures ---
@pytest.fixture
def temp_path():
	path = tempfile.mkdtemp()
	yield path
	if os.path.exists(path):
		shutil.rmtree(path)


@pytest.fixture(scope="session")
def bench_data_100k():
	"""Pre-generates encoded bytes once per session."""
	# Data has repetitive 'C' tail to test compression effectiveness
	val = b"CC(C)C(=O)O" + b"C" * 100
	return [(f"key_{i:08d}".encode(), val) for i in range(100_000)]


# --- Benchmarks ---
@pytest.mark.benchmark(group="write")
@pytest.mark.parametrize("compression", ["none", "snappy", "zlib", "lz4", "lz4hc", "zstd"])
def test_bench_rocksdb_write(benchmark, temp_path, bench_data_100k, compression):
	"""Benchmarks RocksDB write performance with explicit flushing for size measurement."""
	db = PyRockDB(temp_path, read_only=False, compression=compression)

	def run_insert():
		db.put_batch(bench_data_100k)
		db.flush()  # Ensure data moves from WAL/MemTable to SST for size accuracy

	benchmark.pedantic(run_insert, rounds=5, iterations=1)
	db.close()

	size = get_dir_size_rocksdb(temp_path)
	print(f"\nRocksDB [{compression}] Size: {format_size(size)}")


@pytest.mark.benchmark(group="write")
@pytest.mark.parametrize("mode", ["raw", "pickle", "pickle_compressed"])
def test_bench_lmdb_write_optimized(benchmark, temp_path, bench_data_100k, mode):
	"""
	Benchmarks LMDB using the optimized flags from script 2.
	Includes comparison between raw bytes, pickle, and compressed pickle.
	"""
	map_size = 10 * 1024**3  # 10GB
	env = lmdb.open(temp_path, map_size=map_size, writemap=True, map_async=True, sync=False)

	def run_insert():
		with env.begin(write=True) as txn:
			for k, v in bench_data_100k:
				if mode == "raw":
					txn.put(k, v)
				elif mode == "pickle":
					txn.put(k, pickle.dumps(v))
				elif mode == "pickle_compressed":
					# Note: compression overhead happens within the transaction loop
					txn.put(k, zlib.compress(pickle.dumps(v)))
		env.sync()

	benchmark.pedantic(run_insert, rounds=5, iterations=1)

	physical_size = get_actual_lmdb_size(temp_path)
	print(f"\nLMDB [{mode}] Size: {format_size(physical_size)}")
	env.close()


@pytest.mark.benchmark(group="read")
def test_bench_rocksdb_read(benchmark, temp_path, bench_data_100k):
	"""Standard RocksDB read benchmark (DB opened read-only for measurement)."""

	writer_db = PyRockDB(temp_path, read_only=False, compression="lz4")
	writer_db.put_batch(bench_data_100k)
	writer_db.flush()
	writer_db.close()

	db = PyRockDB(temp_path, read_only=True, compression="lz4")
	keys = [k for k, v in bench_data_100k]

	def run_read():
		for k in keys:
			db.get(k)

	benchmark.pedantic(run_read, rounds=3, iterations=1)
	db.close()


@pytest.mark.benchmark(group="read")
def test_bench_lmdb_read(benchmark, temp_path, bench_data_100k):
	"""LMDB reads."""
	env = lmdb.open(temp_path, map_size=10 * 1024**3, writemap=True, map_async=True, sync=False)
	with env.begin(write=True) as txn:
		for k, v in bench_data_100k:
			txn.put(k, v)
	env.sync()

	keys = [k for k, v in bench_data_100k]

	def run_read():
		with env.begin() as txn:
			for k in keys:
				txn.get(k)

	benchmark.pedantic(run_read, rounds=3, iterations=1)
	env.close()


@pytest.mark.benchmark(group="read_batch")
def test_bench_rocksdb_read_batch(benchmark, temp_path, bench_data_100k):
	"""RocksDB batch reads - much faster than individual gets."""
	writer_db = PyRockDB(temp_path, read_only=False, compression="lz4")
	writer_db.put_batch(bench_data_100k)
	writer_db.flush()
	writer_db.close()

	db = PyRockDB(temp_path, read_only=True, compression="lz4")
	keys = [k for k, v in bench_data_100k]

	batch_size = 10_000
	key_batches = [keys[i : i + batch_size] for i in range(0, len(keys), batch_size)]

	def run_read():
		for batch in key_batches:
			db.get_batch(batch)

	benchmark.pedantic(run_read, rounds=3, iterations=1)
	db.close()


@pytest.mark.benchmark(group="read_batch")
def test_bench_lmdb_read_batch_cursor(benchmark, temp_path, bench_data_100k):
	"""LMDB using cursor for batch reads - faster than individual gets."""
	env = lmdb.open(temp_path, map_size=10 * 1024**3, writemap=True, map_async=True, sync=False)
	with env.begin(write=True) as txn:
		for k, v in bench_data_100k:
			txn.put(k, v)
	env.sync()

	def run_read():
		with env.begin() as txn:
			cursor = txn.cursor()
			# Iterate through all keys
			for key, value in cursor:
				pass

	benchmark.pedantic(run_read, rounds=3, iterations=1)
	env.close()


@pytest.mark.benchmark(group="read_pattern")
def test_bench_rocksdb_sequential_read(benchmark, temp_path, bench_data_100k):
	"""Test sequential read pattern (cache-friendly)."""
	db = PyRockDB(temp_path, read_only=False, compression="lz4")
	db.put_batch(bench_data_100k)
	db.flush()
	keys = [k for k, v in bench_data_100k]

	def run_read():
		for k in keys:
			db.get(k)

	benchmark.pedantic(run_read, rounds=3, iterations=1)
	db.close()


@pytest.mark.benchmark(group="read_pattern")
def test_bench_rocksdb_random_read(benchmark, temp_path, bench_data_100k):
	"""Test random read pattern (cache-unfriendly)."""
	import random

	writer_db = PyRockDB(temp_path, read_only=False, compression="lz4")
	writer_db.put_batch(bench_data_100k)
	writer_db.flush()
	writer_db.close()

	db = PyRockDB(temp_path, read_only=True, compression="lz4")
	keys = [k for k, v in bench_data_100k]
	random.shuffle(keys)

	def run_read():
		for k in keys:
			db.get(k)

	benchmark.pedantic(run_read, rounds=3, iterations=1)
	db.close()
