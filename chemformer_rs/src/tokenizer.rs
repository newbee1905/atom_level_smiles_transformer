use pyo3::prelude::*;
use regex::Regex;
use lazy_regex::{lazy_regex, Lazy};
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use rayon::prelude::*;
use ahash::AHashSet;
use rocksdb::{DB, Options, IteratorMode, Direction, DBCompressionType};
use serde::{Serialize, Deserialize};
use serde_yaml;
use std::fs;

/*
* PATCH_REGEX:
* Atoms in brackets: \[[^\]]+\] (e.g., [NH4+], [13C]) - Highest priority
* Special Tags: <[^>]+> (e.g., <UNK>, <RX_1>)
* Elements (Greedy match):
* - Two-letter elements first (e.g., "Cl", "Br", "Au") via character classes.
* - Single-letter elements/specifics (U, V, W, Xe, Y, Zn, Zr, etc).
* Aromatic symbols: b, c, n, o, p, s, se, te
* Special Symbols: [-=#$():+.\\\/~@?><*%]
* Single Digits: \d (0-9)
*/
static MASTER_REGEX: Lazy<Regex> = lazy_regex!(r"(?x)
	(
		\[[^\]]+\] |
		<[^>]+> |
		A[cglmrstu] |
		B[aehikr]?  |
		C[adeflmnorsu]? |
		D[bsy] | 
		E[rsu] |
		F[elmr]? |
		G[ade] |
		H[efgos]? | 
		I[nr]? | 
		K[r]? | 
		L[airuv] |
		M[dgnot] |
		N[abdeiop]? |
		O[gs]? |
		P[abdmortu]? |
		R[abefghnu] |
		S[bcdeimnr]? |
		T[abcehilms] |
		U | V | W | Xe |
		Y[b]? |
		Z[nr] |
		b | c | n | o | p | s[e]? | te |
		- | = | \# | \$ | \( | \) | : | \+ | \. | / | \\ | ~ | @ | \? | > | < | \* | % |
		\d
	)"
);


lazy_static! {
	static ref SMILES_EXPLICIT_TOKENS: Vec<&'static str> = vec![
		// Organic Subset (Common)
		"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", 
		
		// Aromatic Symbols
		"b", "c", "n", "o", "p", "s", "se", "te",

		// Complete Periodic Table
		"Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au",
		"Ba", "Be", "Bh", "Bi", "Bk",
		"Ca", "Cd", "Ce", "Cf", "Cm", "Cn", "Co", "Cr", "Cs", "Cu",
		"Db", "Ds", "Dy",
		"Er", "Es", "Eu",
		"Fe", "Fl", "Fm", "Fr",
		"Ga", "Gd", "Ge",
		"H", "He", "Hf", "Hg", "Ho", "Hs",
		"In", "Ir",
		"K", "Kr",
		"La", "Li", "Lr", "Lu", "Lv",
		"Md", "Mg", "Mn", "Mo", "Mt",
		"Na", "Nb", "Nd", "Ne", "Ni", "No", "Np",
		"Og", "Os",
		"Pa", "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu",
		"Ra", "Rb", "Re", "Rf", "Rg", "Rh", "Rn", "Ru",
		"Sb", "Sc", "Se", "Sg", "Si", "Sm", "Sn", "Sr",
		"Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm", "Ts",
		"U", "V", "W", "Xe", "Y", "Yb", "Zn", "Zr",

		// Special Symbols
		"=", "#", "-", "+", "\\", "/", "(", ")", ".", "@", "?", ">", "<", "*", "%", "$", ":", "~",

		// Ring Digits
		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
	];

	static ref SPECIAL_TOKENS: Vec<String> = vec![
		"<UNK>".to_string(),
		"<PAD>".to_string(),
		"<BOS>".to_string(),
		"<EOS>".to_string(),
		"<MASK>".to_string(),
	];
}

/// Vocabulary data structure for JSON serialization
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VocabularyData {
	pub tokens: Vec<String>,
	pub metadata: VocabMetadata,
}

/// Metadata for vocabulary
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VocabMetadata {
	pub total_tokens: usize,
	pub unique_tokens: usize,
	pub total_smiles_processed: usize,
	pub created_date: String,
	pub source: String,  // e.g., "ZINC", "ChEMBL", "unified_all_datasets"
	pub charset: String, // e.g., "SMILES", "reactions"
}

/// SMILES Tokenizer with JSON vocabulary support
#[pyclass]
pub struct SMILESTokenizer {
	vocabulary: AHashSet<String>,
	stoi: HashMap<String, usize>,
	itos: HashMap<usize, String>,
	unk_token: String,
	bos_token: String,
	eos_token: String,
	pad_token: String,
}

#[pymethods]
impl SMILESTokenizer {
	#[new]
	fn new(vocabulary: Option<Vec<String>>) -> Self {
		pyo3::prepare_freethreaded_python();

		let special_tokens = SPECIAL_TOKENS.iter();
		let explicit_smiles_tokens = SMILES_EXPLICIT_TOKENS.iter().map(|s| s.to_string());

		let mut vocab_set: AHashSet<String> = if let Some(vocab) = vocabulary {
			vocab.into_iter().collect()
		} else {
			AHashSet::new()
		};

		for token in special_tokens {
			vocab_set.insert(token.clone());
		}

		for token in explicit_smiles_tokens {
			vocab_set.insert(token);
		}

		let mut sorted_vocab: Vec<String> = vocab_set.iter().cloned().collect();
		sorted_vocab.sort();

		let stoi: HashMap<String, usize> = sorted_vocab
			.iter()
			.enumerate()
			.map(|(i, token)| (token.clone(), i))
			.collect();

		let itos: HashMap<usize, String> = stoi
			.iter()
			.map(|(token, idx)| (*idx, token.clone()))
			.collect();

		SMILESTokenizer {
			vocabulary: vocab_set,
			stoi,
			itos,
			unk_token: "<UNK>".to_string(),
			bos_token: "<BOS>".to_string(),
			eos_token: "<EOS>".to_string(),
			pad_token: "<PAD>".to_string(),
		}
	}

	/// Load tokenizer from a vocabulary file (JSON or YAML)
	#[staticmethod]
	fn from_vocab(file_path: &str) -> PyResult<SMILESTokenizer> {
		let path = Path::new(file_path);
		match path.extension().and_then(|s| s.to_str()) {
			Some("json") => SMILESTokenizer::from_vocab_json(file_path),
			Some("yaml") | Some("yml") => SMILESTokenizer::from_vocab_yaml(file_path),
			_ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
				"Unsupported file type. Please use .json or .yaml/.yml",
			)),
		}
	}

	/// Load tokenizer from JSON vocabulary file
	#[staticmethod]
	fn from_vocab_json(file_path: &str) -> PyResult<SMILESTokenizer> {
		let content = std::fs::read_to_string(file_path)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to read vocab file: {}", e)
			))?;

		let vocab_data: VocabularyData = serde_json::from_str(&content)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
				format!("Failed to parse JSON: {}", e)
			))?;

		Ok(SMILESTokenizer::new(Some(vocab_data.tokens)))
	}

	/// Load tokenizer from YAML vocabulary file
	#[staticmethod]
	fn from_vocab_yaml(file_path: &str) -> PyResult<SMILESTokenizer> {
		let content = std::fs::read_to_string(file_path)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to read vocab file: {}", e)
			))?;

		let vocab_data: VocabularyData = serde_yaml::from_str(&content)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
				format!("Failed to parse YAML: {}", e)
			))?;

		Ok(SMILESTokenizer::new(Some(vocab_data.tokens)))
	}

	/// Convert JSON vocabulary to YAML format
	#[staticmethod]
	fn json_to_yaml_vocab(json_path: &str, yaml_path: &str) -> PyResult<()> {
		let content = std::fs::read_to_string(json_path)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to read JSON file: {}", e)
			))?;

		let vocab_data: VocabularyData = serde_json::from_str(&content)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
				format!("Failed to parse JSON: {}", e)
			))?;

		let yaml_str = serde_yaml::to_string(&vocab_data)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
				format!("Failed to serialize to YAML: {}", e)
			))?;

		std::fs::write(yaml_path, yaml_str)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to write YAML file: {}", e)
			))?;

		Ok(())
	}

	/// Save vocabulary to JSON file
	fn save_vocab_json(&self, file_path: &str, source: &str) -> PyResult<()> {
		let mut vocab_list: Vec<String> = self.vocabulary.iter().cloned().collect();
		vocab_list.sort();

		let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

		let vocab_data = VocabularyData {
			tokens: vocab_list.clone(),
			metadata: VocabMetadata {
				total_tokens: 0,  // Would need to track separately
				unique_tokens: vocab_list.len(),
				total_smiles_processed: 0,
				created_date: now,
				source: source.to_string(),
				charset: "SMILES".to_string(),
			},
		};

		let json_str = serde_json::to_string_pretty(&vocab_data)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
				format!("Failed to serialize: {}", e)
			))?;

		std::fs::write(file_path, json_str)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to write vocab file: {}", e)
			))?;

		Ok(())
	}

	/// Save vocabulary to YAML file
	fn save_vocab_yaml(&self, file_path: &str, source: &str) -> PyResult<()> {
		let mut vocab_list: Vec<String> = self.vocabulary.iter().cloned().collect();
		vocab_list.sort();

		let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

		let vocab_data = VocabularyData {
			tokens: vocab_list.clone(),
			metadata: VocabMetadata {
				total_tokens: 0,  // Would need to track separately
				unique_tokens: vocab_list.len(),
				total_smiles_processed: 0,
				created_date: now,
				source: source.to_string(),
				charset: "SMILES".to_string(),
			},
		};

		let yaml_str = serde_yaml::to_string(&vocab_data)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
				format!("Failed to serialize to YAML: {}", e)
			))?;

		std::fs::write(file_path, yaml_str)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to write vocab file: {}", e)
			))?;

		Ok(())
	}

	#[getter]
	fn vocab_size(&self) -> usize {
		self.vocabulary.len()
	}

	#[getter]
	fn tokens(&self) -> Vec<String> {
		let mut tokens: Vec<String> = self.vocabulary.iter().cloned().collect();
		tokens.sort();
		tokens
	}

	fn _tokenize_regex(&self, smiles: &str) -> Vec<String> {
		MASTER_REGEX
			.find_iter(smiles)
			.map(|m| m.as_str().to_string())
			.collect()
	}

	#[pyo3(signature = (smiles, add_bos=true, add_eos=true, max_length=None, pad_to_length=false))]
	fn tokenize(
		&self,
		smiles: &str,
		add_bos: bool,
		add_eos: bool,
		max_length: Option<usize>,
		pad_to_length: bool,
	) -> (Vec<String>, Vec<bool>) {
		let mut tokens = self._tokenize_regex(smiles);

		tokens = tokens
			.into_iter()
			.map(|token| {
				if self.vocabulary.contains(&token) {
					token
				} else {
					self.unk_token.clone()
				}
			})
			.collect();

		if add_bos {
			tokens.insert(0, self.bos_token.clone());
		}
		if add_eos {
			tokens.push(self.eos_token.clone());
		}

		let mut mask = vec![true; tokens.len()];

		if let Some(max_len) = max_length {
			if tokens.len() > max_len {
				tokens.truncate(max_len);
				mask.truncate(max_len);
			} else if pad_to_length && tokens.len() < max_len {
				let pad_count = max_len - tokens.len();
				for _ in 0..pad_count {
					tokens.push(self.pad_token.clone());
					mask.push(false);
				}
			}
		}

		(tokens, mask)
	}

	#[pyo3(signature = (smiles, add_bos=true, add_eos=true, max_length=None, pad_to_length=false))]
	fn encode(
		&self,
		smiles: &str,
		add_bos: bool,
		add_eos: bool,
		max_length: Option<usize>,
		pad_to_length: bool,
	) -> (Vec<usize>, Vec<bool>) {
		let (tokens, mask) = self.tokenize(smiles, add_bos, add_eos, max_length, pad_to_length);

		let indices: Vec<usize> = tokens
			.into_iter()
			.map(|token| {
				*self
					.stoi
					.get(&token)
					.unwrap_or_else(|| self.stoi.get(&self.unk_token).unwrap())
			})
			.collect();

		(indices, mask)
	}

	fn decode(&self, indices: Vec<usize>) -> Vec<String> {
		indices
			.into_iter()
			.map(|idx| {
				self.itos
					.get(&idx)
					.cloned()
					.unwrap_or_else(|| self.unk_token.clone())
			})
			.collect()
	}

	fn decode_to_string(&self, indices: Vec<usize>) -> String {
		self.decode(indices).join(" ")
	}

	fn token_to_index(&self, token: &str) -> Option<usize> {
		self.stoi.get(token).copied()
	}

	fn index_to_token(&self, idx: usize) -> Option<String> {
		self.itos.get(&idx).cloned()
	}

}

/// Static helper function for tokenization
fn tokenize_regex(smiles: &str) -> Vec<String> {
	MASTER_REGEX
		.find_iter(smiles)
		.map(|m| m.as_str().to_string())
		.collect()
}

/// RocksDB Vocabulary Builder with JSON export
#[pyclass]
pub struct RocksDBVocabBuilder {
	db_path: String,
	chunk_size: usize,
}

#[pymethods]
impl RocksDBVocabBuilder {
	#[new]
	fn new(db_path: String, chunk_size: usize) -> Self {
		RocksDBVocabBuilder { db_path, chunk_size }
	}

	/// Build vocabulary from SMILES file and optionally export to JSON
	fn build_from_file(
		&self,
		file_path: &str,
		num_threads: Option<usize>,
		export_json: Option<&str>,
	) -> PyResult<(Vec<String>, Vec<(String, usize)>, (usize, usize, usize))> {
		let _num_threads = num_threads.unwrap_or_else(num_cpus::get);

		fs::create_dir_all(&self.db_path)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to create DB path: {}", e)
			))?;

		// Initialize RocksDB with optimized options
		let mut opts = Options::default();
		opts.create_if_missing(true);
		opts.set_compression_type(DBCompressionType::Lz4);

		let db = DB::open(&opts, &self.db_path)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to open RocksDB: {}", e)
			))?;

		let token_counts: Arc<parking_lot::Mutex<HashMap<String, usize>>> = 
			Arc::new(parking_lot::Mutex::new(HashMap::new()));

		let content = std::fs::read_to_string(file_path)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to read file: {}", e)
			))?;

		let lines: Vec<&str> = content.lines().collect();
		let total_smiles = lines.len();

		// Process in parallel chunks
		let chunk_results: Vec<_> = lines
			.par_chunks(self.chunk_size)
			.map(|chunk| {
				let mut local_counts: HashMap<String, usize> = HashMap::new();

				for smiles in chunk {
					let tokens = tokenize_regex(smiles);
					for token in tokens {
						*local_counts.entry(token).or_insert(0) += 1;
					}
				}

				local_counts
			})
			.collect();

		// Merge all chunk results
		let mut total_tokens = 0usize;
		{
			let mut tc = token_counts.lock();
			for local_counts in chunk_results {
				for (token, count) in local_counts {
					*tc.entry(token).or_insert(0) += count;
					total_tokens += count;
				}
			}
		}

		// Write to RocksDB using batch write for efficiency
		let mut batch = rocksdb::WriteBatch::default();
		{
			let tc = token_counts.lock();
			for (token, count) in tc.iter() {
				let count_bytes = bincode::serialize(count)
					.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
						format!("Serialization failed: {}", e)
					))?;

				batch.put(token.as_bytes(), &count_bytes);
			}
		}

		db.write(batch)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to write batch to RocksDB: {}", e)
			))?;

		// Read vocabulary back and sort
		let mut sorted_vocab = Vec::new();
		{
			let iter = db.iterator(IteratorMode::From(b"", Direction::Forward));
			for result in iter {
				match result {
					Ok((key, _)) => {
						if let Ok(key_str) = std::str::from_utf8(&key) {
							sorted_vocab.push(key_str.to_string());
						}
					}
					Err(e) => {
						return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
							format!("Iterator error: {}", e)
						));
					}
				}
			}
		}

		// Add special tokens
		let special_tokens = vec![
			"<UNK>".to_string(),
			"<PAD>".to_string(),
			"<SOS>".to_string(),
			"<EOS>".to_string(),
		];

		for token in special_tokens {
			if !sorted_vocab.contains(&token) {
				sorted_vocab.push(token);
			}
		}

		for token in SMILES_EXPLICIT_TOKENS.iter() {
			let t = token.to_string();
			if !sorted_vocab.contains(&t) {
				sorted_vocab.push(t);
			}
		}

		sorted_vocab.sort();
		let unique_tokens = sorted_vocab.len();

		// Export to JSON if requested
		if let Some(json_path) = export_json {
			let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
			let vocab_data = VocabularyData {
				tokens: sorted_vocab.clone(),
				metadata: VocabMetadata {
					total_tokens,
					unique_tokens,
					total_smiles_processed: total_smiles,
					created_date: now,
					source: "from_file".to_string(),
					charset: "SMILES".to_string(),
				},
			};

			let json_str = serde_json::to_string_pretty(&vocab_data)
				.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
					format!("JSON serialization failed: {}", e)
				))?;

			std::fs::write(json_path, json_str)
				.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
					format!("Failed to write JSON: {}", e)
				))?;
		}

		// Get top 10 tokens by frequency
		let tc = token_counts.lock();
		let mut token_freq: Vec<(String, usize)> = tc
			.iter()
			.map(|(token, count)| (token.clone(), *count))
			.collect();

		token_freq.sort_by(|a, b| b.1.cmp(&a.1));
		let top_tokens: Vec<(String, usize)> = token_freq.into_iter().take(10).collect();

		Ok((sorted_vocab, top_tokens, (total_tokens, unique_tokens, total_smiles)))
	}

	/// Build vocabulary from iterator and optionally export to JSON
	fn build_from_iterator(
		&self,
		smiles_iter: Vec<String>,
		num_threads: Option<usize>,
		export_json: Option<&str>,
	) -> PyResult<(Vec<String>, Vec<(String, usize)>, (usize, usize, usize))> {
		let _num_threads = num_threads.unwrap_or_else(num_cpus::get);

		fs::create_dir_all(&self.db_path)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to create DB path: {}", e)
			))?;

		// Initialize RocksDB with optimized options
		let mut opts = Options::default();
		opts.create_if_missing(true);
		opts.set_compression_type(DBCompressionType::Lz4);

		let db = DB::open(&opts, &self.db_path)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to open RocksDB: {}", e)
			))?;

		let token_counts: Arc<parking_lot::Mutex<HashMap<String, usize>>> = 
			Arc::new(parking_lot::Mutex::new(HashMap::new()));

		let total_smiles = smiles_iter.len();

		// Process in parallel chunks
		let _: Vec<_> = smiles_iter
			.par_chunks(self.chunk_size)
			.map(|chunk| {
				let mut local_counts: HashMap<String, usize> = HashMap::new();

				for smiles in chunk {
					let tokens = tokenize_regex(smiles);
					for token in tokens {
						*local_counts.entry(token).or_insert(0) += 1;
					}
				}

				let mut tc = token_counts.lock();
				for (token, count) in local_counts {
					*tc.entry(token).or_insert(0) += count;
				}
			})
			.collect();

		// Write to RocksDB using batch write
		let mut batch = rocksdb::WriteBatch::default();
		let mut total_tokens = 0usize;
		{
			let tc = token_counts.lock();
			for (token, count) in tc.iter() {
				total_tokens += count;
				let count_bytes = bincode::serialize(count)
					.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
						format!("Serialization failed: {}", e)
					))?;

				batch.put(token.as_bytes(), &count_bytes);
			}
		}

		db.write(batch)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to write batch to RocksDB: {}", e)
			))?;

		// Read vocabulary and sort
		let mut sorted_vocab = Vec::new();
		{
			let iter = db.iterator(IteratorMode::From(b"", Direction::Forward));
			for result in iter {
				match result {
					Ok((key, _)) => {
						if let Ok(key_str) = std::str::from_utf8(&key) {
							sorted_vocab.push(key_str.to_string());
						}
					}
					Err(e) => {
						return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
							format!("Iterator error: {}", e)
						));
					}
				}
			}
		}

		// Add special tokens
		let special_tokens = vec![
			"<UNK>".to_string(),
			"<PAD>".to_string(),
			"<SOS>".to_string(),
			"<EOS>".to_string(),
		];

		for token in special_tokens {
			if !sorted_vocab.contains(&token) {
				sorted_vocab.push(token);
			}
		}

		sorted_vocab.sort();
		let unique_tokens = sorted_vocab.len();

		// Export to JSON if requested
		if let Some(json_path) = export_json {
			let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
			let vocab_data = VocabularyData {
				tokens: sorted_vocab.clone(),
				metadata: VocabMetadata {
					total_tokens,
					unique_tokens,
					total_smiles_processed: total_smiles,
					created_date: now,
					source: "from_iterator".to_string(),
					charset: "SMILES".to_string(),
				},
			};

			let json_str = serde_json::to_string_pretty(&vocab_data)
				.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
					format!("JSON serialization failed: {}", e)
				))?;

			std::fs::write(json_path, json_str)
				.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
					format!("Failed to write JSON: {}", e)
				))?;
		}

		// Get top 10 tokens
		let tc = token_counts.lock();
		let mut token_freq: Vec<(String, usize)> = tc
			.iter()
			.map(|(token, count)| (token.clone(), *count))
			.collect();

		token_freq.sort_by(|a, b| b.1.cmp(&a.1));
		let top_tokens: Vec<(String, usize)> = token_freq.into_iter().take(10).collect();

		Ok((sorted_vocab, top_tokens, (total_tokens, unique_tokens, total_smiles)))
	}

	/// Load vocabulary from RocksDB
	fn load_vocab_from_db(&self) -> PyResult<Vec<String>> {
		let mut opts = Options::default();
		opts.create_if_missing(false);

		let db = DB::open_for_read_only(&opts, &self.db_path, false)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to open RocksDB: {}", e)
			))?;

		let mut vocab = Vec::new();
		let iter = db.iterator(IteratorMode::From(b"", Direction::Forward));

		for result in iter {
			match result {
				Ok((key, _)) => {
					if let Ok(key_str) = std::str::from_utf8(&key) {
						vocab.push(key_str.to_string());
					}
				}
				Err(e) => {
					return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
						format!("Iterator error: {}", e)
					));
				}
			}
		}

		vocab.sort();
		Ok(vocab)
	}

	/// Get token frequency from RocksDB
	fn get_token_frequency(&self, token: &str) -> PyResult<usize> {
		let mut opts = Options::default();
		opts.create_if_missing(false);

		let db = DB::open_for_read_only(&opts, &self.db_path, false)
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("Failed to open RocksDB: {}", e)
			))?;

		match db.get(token.as_bytes())
			.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
				format!("RocksDB error: {}", e)
			))?
		{
			Some(value_bytes) => {
				bincode::deserialize::<usize>(&value_bytes)
					.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
						format!("Deserialization failed: {}", e)
					))
			}
			None => Ok(0),
		}
	}

	/// Clear database
	fn clear_db(&self) -> PyResult<()> {
		if Path::new(&self.db_path).exists() {
			fs::remove_dir_all(&self.db_path)
				.map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(
					format!("Failed to clear DB: {}", e)
				))?;
		}
		Ok(())
	}
}

/// Legacy function for backward compatibility
#[pyfunction]
pub fn build_vocabulary(smiles_list: Vec<String>) -> (Vec<String>, Vec<(String, usize)>) {
	let special_tokens = vec![
		"<UNK>".to_string(),
		"<PAD>".to_string(),
		"<SOS>".to_string(),
		"<EOS>".to_string(),
	];

	let all_tokens: Vec<String> = smiles_list
		.par_iter()
		.flat_map(|smiles| {
			MASTER_REGEX
				.find_iter(smiles)
				.map(|m| m.as_str().to_string())
				.collect::<Vec<_>>()
		})
		.collect();

	let token_counts: HashMap<String, usize> = all_tokens
		.iter()
		.fold(HashMap::new(), |mut acc, token| {
			*acc.entry(token.clone()).or_insert(0) += 1;
			acc
		});

	let mut vocabulary: AHashSet<String> = all_tokens.into_iter().collect();
	for token in &special_tokens {
		vocabulary.insert(token.clone());
	}

	let mut sorted_vocab: Vec<String> = vocabulary.iter().cloned().collect();
	sorted_vocab.sort();

	let mut token_freq: Vec<(String, usize)> = token_counts.into_iter().collect();
	token_freq.sort_by(|a, b| b.1.cmp(&a.1));
	let top_tokens: Vec<(String, usize)> = token_freq.into_iter().take(10).collect();

	(sorted_vocab, top_tokens)
}
