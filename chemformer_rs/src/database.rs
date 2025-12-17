use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rocksdb::{DB, Options, DBCompressionType, WriteBatch};
use std::sync::Arc;

#[pyclass]
pub struct PyRockDB {
	// DB is Sync + Send, wrapping in Arc for cheap cloning across threads if needed
	db: Arc<DB>,
}

#[pymethods]
impl PyRockDB {
	/// Open a RocksDB database.
	///
	/// Args:
	///	 path (str): File system path to the database.
	///	 read_only (bool): If True, opens in read-only mode. Default is False.
	#[new]
	#[pyo3(signature = (path, read_only=false))]
	fn new(path: &str, read_only: bool) -> PyResult<Self> {
		let mut opts = Options::default();
		opts.create_if_missing(true);
		// Use LZ4 compression as it is fast and efficient for text/JSON data
		opts.set_compression_type(DBCompressionType::Lz4);

		let db = if read_only {
			DB::open_for_read_only(&opts, path, false).map_err(|e| {
				PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open DB: {}", e))
			})?
		} else {
			DB::open(&opts, path).map_err(|e| {
				PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open DB: {}", e))
			})?
		};

		Ok(PyRockDB { db: Arc::new(db) })
	}

	/// Put a key-value pair into the database.
	///
	/// Releases GIL during the write operation.
	fn put(&self, py: Python, key: &[u8], value: &[u8]) -> PyResult<()> {
		let db = self.db.clone();
		let k = key.to_vec();
		let v = value.to_vec();

		// release_gil allows other Python threads to run while RocksDB writes
		py.allow_threads(move || {
			db.put(&k, &v).map_err(|e| {
				pyo3::exceptions::PyIOError::new_err(format!("Failed to put item: {}", e))
			})
		})
	}

	/// Get a value by key.
	///
	/// Releases GIL during the read operation.
	/// Returns: bytes or None
	fn get<'a>(&self, py: Python<'a>, key: &[u8]) -> PyResult<Option<Bound<'a, PyBytes>>> {
		let db = self.db.clone();
		let k = key.to_vec();

		let result = py.allow_threads(move || {
			db.get(&k).map_err(|e| {
				pyo3::exceptions::PyIOError::new_err(format!("Failed to get item: {}", e))
			})
		})?;

		match result {
			Some(bytes) => Ok(Some(PyBytes::new_bound(py, &bytes))),
			None => Ok(None),
		}
	}

	/// Write a batch of key-value pairs.
	///
	/// Args:
  ///	 data (List[Tuple[bytes, bytes]]): A list of (key, value) tuples.
	fn put_batch(&self, py: Python, data: Vec<(Vec<u8>, Vec<u8>)>) -> PyResult<()> {
		let db = self.db.clone();
		// We must own the data to move it into the thread closure
		let owned_data: Vec<(Vec<u8>, Vec<u8>)> = data.iter()
			.map(|(k, v)| (k.to_vec(), v.to_vec()))
			.collect();

		py.allow_threads(move || {
			let mut batch = WriteBatch::default();
			for (k, v) in owned_data {
				batch.put(&k, &v);
			}
			db.write(batch).map_err(|e| {
				pyo3::exceptions::PyIOError::new_err(format!("Batch write failed: {}", e))
			})
		})
	}

	/// Close the database connection manually.
	///
	/// Note: This is a no-op in this implementation. Resources are automatically 
	/// released when the object is garbage collected.
	fn close(&self) {
		// No explicit action needed; RAII handles cleanup.
	}
}
