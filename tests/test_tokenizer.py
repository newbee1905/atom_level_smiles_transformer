import pytest
import shutil
import tempfile
import os
from pathlib import Path
from chemformer_rs import SMILESTokenizer, RocksDBVocabBuilder

@pytest.fixture
def temp_path():
	"""Creates a temporary directory and cleans up after."""
	path = tempfile.mkdtemp()
	yield path
	shutil.rmtree(path)

@pytest.fixture
def tokenizer():
	"""Returns a tokenizer instance with default settings."""
	return SMILESTokenizer(None)

def test_tokenizer_regex_splitting(tokenizer):
	smiles = "CC(=O)N"
	# Expected: C, C, (, =, O, ), N (plus BOS/EOS if enabled by default)
	tokens, mask = tokenizer.tokenize(smiles, add_bos=False, add_eos=False)
	assert tokens == ['C', 'C', '(', '=', 'O', ')', 'N']

def test_tokenizer_encode_decode_roundtrip(tokenizer):
	smiles = "c1ccccc1" # benzene
	
	indices, mask = tokenizer.encode(smiles, add_bos=True, add_eos=True)
	decoded_tokens = tokenizer.decode(indices)
	
	assert decoded_tokens[0] == "<BOS>"
	assert decoded_tokens[-1] == "<EOS>"
	
	content = "".join(decoded_tokens[1:-1])
	assert content == smiles

def test_tokenizer_padding(tokenizer):
	smiles = "C"
	max_len = 5
	indices, mask = tokenizer.encode(smiles, max_length=max_len, pad_to_length=True)
	
	assert len(indices) == max_len

	# Assuming <PAD> is index 1 (or specifically checked via token_to_index)
	pad_idx = tokenizer.token_to_index("<PAD>")
	assert indices[-1] == pad_idx
	assert mask[-1] is False

def test_vocab_builder(temp_path):
	smiles_file = Path(temp_path) / "input.smi"
	with open(smiles_file, "w") as f:
		f.write("CCO\n")
		f.write("CCN\n")
		f.write("CCO\n") 
	
	builder_db_path = os.path.join(temp_path, "vocab_build")
	builder = RocksDBVocabBuilder(builder_db_path, chunk_size=1024)
	
	vocab, top_tokens, stats = builder.build_from_file(str(smiles_file), num_threads=1, export_json=None)
	assert "C" in vocab
	
	total_tokens, unique_tokens, total_smiles = stats
	assert total_smiles == 3
