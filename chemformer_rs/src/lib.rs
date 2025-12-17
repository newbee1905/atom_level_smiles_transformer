use pyo3::prelude::*;

mod tokenizer;
mod database;

use tokenizer::{SMILESTokenizer, RocksDBVocabBuilder, build_vocabulary};
use database::PyRockDB;

#[pymodule]
fn chemformer_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SMILESTokenizer>()?;
    m.add_class::<RocksDBVocabBuilder>()?;
    m.add_function(wrap_pyfunction!(build_vocabulary, m)?)?;
    m.add_class::<PyRockDB>()?;
    Ok(())
}
