use pyo3::prelude::*;

mod tokenizer;
mod database;

use tokenizer::{SMILESTokenizer, RocksDBVocabBuilder, build_vocabulary};
use database::PyRockDB;

// #[pymodule]
// fn chemformer_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
// 	let tokenizer_mod = PyModule::new(py, "tokenizer")?;
// 	tokenizer_mod.add_class::<SMILESTokenizer>()?;
// 	tokenizer_mod.add_class::<RocksDBVocabBuilder>()?;
// 	tokenizer_mod.add_function(wrap_pyfunction!(build_vocabulary, &tokenizer_mod)?)?;
//
// 	let database_mod = PyModule::new(py, "database")?;
// 	database_mod.add_class::<PyRockDB>()?;
//
// 	m.add_submodule(&tokenizer_mod)?;
// 	m.add_submodule(&database_mod)?;
//
// 	Ok(())
// }
#[pymodule]
fn chemformer_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
	let tokenizer_mod = PyModule::new_bound(py, "tokenizer")?;
	tokenizer_mod.add_class::<SMILESTokenizer>()?;
	tokenizer_mod.add_class::<RocksDBVocabBuilder>()?;
	tokenizer_mod.add_function(wrap_pyfunction!(build_vocabulary, &tokenizer_mod)?)?;
	
	let database_mod = PyModule::new_bound(py, "database")?;
	database_mod.add_class::<PyRockDB>()?;

	m.add_submodule(&tokenizer_mod)?;
	m.add_submodule(&database_mod)?;

	let sys = py.import_bound("sys")?;
	let modules = sys.getattr("modules")?;
	modules.set_item("chemformer_rs.tokenizer", &tokenizer_mod)?;
	modules.set_item("chemformer_rs.database", &database_mod)?;

	Ok(())
}
