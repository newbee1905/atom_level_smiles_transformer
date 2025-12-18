use pyo3::prelude::*;
use pyo3::types::{PySequence, PyBytes, PyDict};
use rocksdb::{DB, Options, DBCompressionType, WriteBatch, BlockBasedOptions, Cache, DBCompactionStyle, ReadOptions};
use std::sync::Arc;

#[pyclass]
pub struct PyRockDB {
	db: Arc<DB>,
}

#[pymethods]
impl PyRockDB {
	/// Open a RocksDB database
	#[new]
	#[pyo3(signature = (path, read_only=false, compression=None))]
	fn new(path: &str, read_only: bool, compression: Option<&str>) -> PyResult<Self> {
		pyo3::prepare_freethreaded_python();

		let num_cpus = num_cpus::get() as i32;
		let mut opts = Options::default();
		opts.create_if_missing(true);

		opts.increase_parallelism(num_cpus.max(2));
		opts.set_max_background_jobs(num_cpus.clamp(2, 4));

		opts.set_write_buffer_size(256 * 1024 * 1024); 
		opts.set_max_write_buffer_number(4);
		opts.set_min_write_buffer_number_to_merge(1);

		opts.set_optimize_filters_for_hits(true);

		opts.set_compaction_style(DBCompactionStyle::Level);
		opts.set_level_compaction_dynamic_level_bytes(true);
		opts.set_level_zero_file_num_compaction_trigger(4);

		opts.set_level_zero_file_num_compaction_trigger(4);
		opts.set_level_compaction_dynamic_level_bytes(true);

		let mut block_opts = BlockBasedOptions::default();
		// 5123LRU cache is a sensible default for most applications.
		let cache = Cache::new_lru_cache(512 * 1024 * 1024); 
		block_opts.set_block_cache(&cache);
		block_opts.set_block_size(4 * 1024); // 16KB blocks for better random read/write balance
		block_opts.set_cache_index_and_filter_blocks(true);
		block_opts.set_bloom_filter(10.0, false);
		opts.set_block_based_table_factory(&block_opts);

		let mut compression_type = DBCompressionType::None;
		if let Some(c_type) = compression {
			compression_type = match c_type.to_lowercase().as_str() {
				"none" => DBCompressionType::None,
				"snappy" => DBCompressionType::Snappy,
				"zlib" => DBCompressionType::Zlib,
				"bz2" => DBCompressionType::Bz2,
				"lz4" => DBCompressionType::Lz4,
				"lz4hc" => DBCompressionType::Lz4hc,
				"zstd" => DBCompressionType::Zstd,
				_ => return Err(pyo3::exceptions::PyValueError::new_err(
					"Invalid compression: snappy, zlib, bz2, lz4, lz4hc, zstd"
				)),
			};
		} 

		opts.set_compression_type(compression_type);

		let db = if read_only {
			DB::open_for_read_only(&opts, path, false)
		} else {
			DB::open(&opts, path)
		}.map_err(|e| {
			PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open DB: {}", e))
		})?;

		Ok(PyRockDB { 
			db: Arc::new(db),
		})
	}

	fn put(&self, py: Python, key: &[u8], value: &[u8]) -> PyResult<()> {
		let db = self.db.clone();
		py.allow_threads(move || {
			db.put(key, value).map_err(|e| {
				pyo3::exceptions::PyIOError::new_err(format!("Failed to put item: {}", e))
			})
		})
	}

	// fn get<'a>(&self, py: Python<'a>, key: &[u8]) -> PyResult<Option<Bound<'a, PyBytes>>> {
	// 	let db = self.db.clone();
	// 	let result = py.allow_threads(move || {
	// 		db.get(key).map_err(|e| {
	// 			pyo3::exceptions::PyIOError::new_err(format!("Failed to get item: {}", e))
	// 		})
	// 	})?;
	//
	// 	match result {
	// 		Some(bytes) => Ok(Some(PyBytes::new_bound(py, &bytes))),
	// 		None => Ok(None),
	// 	}
	// }

	fn get<'a>(&self, py: Python<'a>, key: &[u8]) -> PyResult<Option<Bound<'a, PyBytes>>> {
		let db = self.db.clone();
		let key = key.to_vec();
		
		let result = py.allow_threads(move || {
			let mut read_opts = ReadOptions::default();
			read_opts.fill_cache(true); 
			read_opts.set_verify_checksums(false);
			
			db.get_opt(&key, &read_opts).map_err(|e| {
				pyo3::exceptions::PyIOError::new_err(format!("Failed to get item: {}", e))
			})
		})?;

		match result {
			Some(bytes) => Ok(Some(PyBytes::new_bound(py, &bytes))),
			None => Ok(None),
		}
	}

	// fn get_batch<'a>(&self, py: Python<'a>, keys: Bound<'_, PySequence>) -> PyResult<Vec<Option<Bound<'a, PyBytes>>>> {
	// 	let len = keys.len()?;
	// 	let mut key_vec = Vec::with_capacity(len);
	//
	// 	// Extract keys while holding GIL
	// 	for i in 0..len {
	// 		let item = keys.get_item(i)?;
	// 		let k: &[u8] = item.extract()?;
	// 		key_vec.push(k.to_vec());
	// 	}
	//
	// 	let db = self.db.clone();
	// 	let results = py.allow_threads(move || {
	// 		let mut read_opts = ReadOptions::default();
	// 		read_opts.fill_cache(true);
	// 		read_opts.set_verify_checksums(false);
	//
	// 		let mut out = Vec::with_capacity(key_vec.len());
	// 		for key in &key_vec {
	// 			match db.get_opt(key, &read_opts) {
	// 				Ok(val) => out.push(val),
	// 				Err(e) => return Err(pyo3::exceptions::PyIOError::new_err(format!("Failed to get item: {}", e))),
	// 			}
	// 		}
	// 		Ok(out)
	// 	})?;
	//
	// 	Ok(results.into_iter().map(|r| {
	// 		r.map(|bytes| PyBytes::new_bound(py, &bytes))
	// 	}).collect())
	// }
	fn get_batch<'a>(&self, py: Python<'a>, keys: Bound<'_, PySequence>) -> PyResult<Bound<'a, PyDict>> {
		let mut key_refs: Vec<Vec<u8>> = Vec::with_capacity(keys.len()?);
		
		// Phase 1: Collect all keys while holding GIL
		for i in 0..keys.len()? {
			let k_obj = keys.get_item(i)?;
			let k: &[u8] = k_obj.extract()?;
			key_refs.push(k.to_vec());
		}

		let db = self.db.clone();

		// Phase 2: Release GIL, do all reads
		let results: Vec<(Vec<u8>, Option<Vec<u8>>)> = py.allow_threads(move || {
			let mut res: Vec<(Vec<u8>, Option<Vec<u8>>)> = Vec::with_capacity(key_refs.len());
			for key in key_refs {
				match db.get(&key) {
					Ok(Some(val)) => res.push((key, Some(val))),
					Ok(None) => res.push((key, None)),
					Err(_) => res.push((key, None)),
				}
			}
			res
		});

		// Phase 3: Return as dict with GIL re-acquired
		let dict = PyDict::new_bound(py);
		for (key, val) in results {
			// Convert Vec<u8> key to PyBytes so Python sees bytes, not list
			let py_key = PyBytes::new_bound(py, &key);
			match val {
				Some(v) => {
					let py_val = PyBytes::new_bound(py, &v);
					dict.set_item(py_key, py_val)?;
				}
				None => {
					dict.set_item(py_key, py.None())?;
				}
			}
		}
		Ok(dict)
	}

	fn delete(&self, py: Python, key: &[u8]) -> PyResult<()> {
		let db = self.db.clone();
		py.allow_threads(move || {
			db.delete(key).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
		})
	}

	fn put_batch(&self, py: Python, data: Bound<'_, PySequence>) -> PyResult<()> {
		let mut batch = WriteBatch::default();

		// Iterate through the sequence manually while holding the GIL.
		for i in 0..data.len()? {
			let item = data.get_item(i)?;
			let tuple: Bound<'_, PySequence> = item.downcast_into().map_err(|_| {
				pyo3::exceptions::PyTypeError::new_err("Batch items must be sequences (tuples/lists) of (key, value)")
			})?;
			
			if tuple.len()? != 2 {
				return Err(pyo3::exceptions::PyValueError::new_err("Each batch item must be a pair of (key, value)"));
			}

			let k_obj = tuple.get_item(0)?;
			let v_obj = tuple.get_item(1)?;

			let k: &[u8] = k_obj.extract()?;
			let v: &[u8] = v_obj.extract()?;
			
			batch.put(k, v);
		}

		let db = self.db.clone();
		py.allow_threads(move || {
			let mut write_opts = rocksdb::WriteOptions::default();
			write_opts.disable_wal(false);
			write_opts.set_sync(false);

			db.write_opt(batch, &write_opts).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
		})
	}

	fn flush(&self, py: Python) -> PyResult<()> {
		let db = self.db.clone();
		py.allow_threads(move || {
			db.flush().map_err(|e| {
				pyo3::exceptions::PyIOError::new_err(format!("Flush failed: {}", e))
			})
		})
	}

	fn compact_range(&self, py: Python) -> PyResult<()> {
		let db = self.db.clone();
		py.allow_threads(move || {
			db.compact_range(None::<&[u8]>, None::<&[u8]>);
			Ok(())
		})
	}

	fn close(&self) {
	}
}
