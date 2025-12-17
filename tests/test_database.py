import pytest
import shutil
import tempfile
import concurrent.futures
from chemformer_rs import PyRockDB

@pytest.fixture
def temp_db_path():
	"""Creates a temporary directory for the RocksDB and cleans up after."""
	path = tempfile.mkdtemp()
	yield path
	shutil.rmtree(path)

def test_db_read_write(temp_db_path):
	db = PyRockDB(temp_db_path, read_only=False)

	key = b"molecule_1"
	value = b"CCO"
	db.put(key, value)
	
	result = db.get(key)
	assert result == value
	
	assert db.get(b"non_existent") is None
	
	db.close()

def test_db_batch_write(temp_db_path):
	db = PyRockDB(temp_db_path, read_only=False)
	
	batch_data = [
		(f"key_{i}".encode('utf-8'), f"val_{i}".encode('utf-8'))
		for i in range(100)
	]
	
	db.put_batch(batch_data)
	
	assert db.get(b"key_0") == b"val_0"
	assert db.get(b"key_99") == b"val_99"
	db.close()

def test_db_concurrency(temp_db_path):
	"""
	Crucial Test: Verifies that Rust releases the GIL.
	If GIL is not released, this might run sequentially or deadlock 
	depending on how heavy the operations are.
	"""
	db = PyRockDB(temp_db_path, read_only=False)
	
	def worker(idx):
		key = f"thread_key_{idx}".encode('utf-8')
		val = f"thread_val_{idx}".encode('utf-8')
		db.put(key, val)
		return db.get(key)

	with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
		futures = [executor.submit(worker, i) for i in range(50)]
		results = [f.result() for f in futures]
	
	for i, res in enumerate(results):
		assert res == f"thread_val_{i}".encode('utf-8')

	db.close()
