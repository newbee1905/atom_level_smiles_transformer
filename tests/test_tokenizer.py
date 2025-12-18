import pytest
import shutil
import tempfile
import os
import json  
from pathlib import Path
from chemformer_rs.tokenizer import SMILESTokenizer, RocksDBVocabBuilder


@pytest.fixture
def temp_path():
	"""Creates a temporary directory and cleans up after."""
	path = tempfile.mkdtemp()
	yield path
	shutil.rmtree(path)


@pytest.fixture
def full_vocab_tokenizer():
	"""Returns a tokenizer instance loaded from a full vocabulary file."""
	vocab_path = Path(__file__).parent.parent / "config" / "vocab.yaml"
	return SMILESTokenizer.from_vocab_yaml(str(vocab_path))


@pytest.fixture
def tokenizer():
	"""Returns a tokenizer instance with default settings (minimal vocab)."""
	return SMILESTokenizer(None)


def test_tokenizer_regex_splitting(tokenizer):
	smiles = "CC(=O)N"

	# Expected: C, C, (, =, O, ), N (plus BOS/EOS if enabled by default)
	tokens, mask = tokenizer.tokenize(smiles, add_bos=False, add_eos=False)
	assert tokens == ["C", "C", "(", "=", "O", ")", "N"]


def test_tokenizer_encode_decode_roundtrip(full_vocab_tokenizer):
	smiles = "c1ccccc1"  # benzene

	indices, mask = full_vocab_tokenizer.encode(smiles, add_bos=True, add_eos=True)
	decoded_tokens = full_vocab_tokenizer.decode(indices)

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


def test_tokenizer_special_characters_and_brackets(full_vocab_tokenizer):
	"""Test tokenization of SMILES with special characters and bracketed atoms."""
	smiles = "C(=O)[NH3+]c1ccccc1"
	tokens, _ = full_vocab_tokenizer.tokenize(smiles, add_bos=False, add_eos=False)
	expected = ["C", "(", "=", "O", ")", "[NH3+]", "c", "1", "c", "c", "c", "c", "c", "1"]
	assert tokens == expected


def test_tokenizer_truncation(full_vocab_tokenizer):
	"""Test tokenizer truncation when max_length is exceeded."""
	smiles = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
	max_len = 10

	# Add 2 for BOS/EOS tokens
	indices, mask = full_vocab_tokenizer.encode(smiles, max_length=max_len, add_bos=True, add_eos=True)
	assert len(indices) == max_len
	assert len(mask) == max_len

	# Check if BOS is present and EOS is NOT (due to truncation)
	assert full_vocab_tokenizer.index_to_token(indices[0]) == "<BOS>"
	assert full_vocab_tokenizer.index_to_token(indices[-1]) != "<EOS>"


def test_tokenizer_unk_token(tokenizer):
	"""Test that unknown tokens are correctly mapped to <UNK>."""
	vocab_list = ["C", "c", "1", "O", "<UNK>", "<BOS>", "<EOS>", "<PAD>"]
	small_vocab_tokenizer = SMILESTokenizer(vocab_list)
	smiles_with_unknown = "C[X]O"  # [X] should be unknown if not in vocab
	tokens_with_unk, _ = small_vocab_tokenizer.tokenize(smiles_with_unknown, add_bos=False, add_eos=False)

	# The regex correctly extracts [X], but it's not in the small_vocab_tokenizer's vocabulary
	# so it should be mapped to <UNK>
	assert tokens_with_unk == ["C", "<UNK>", "O"]


def test_json_to_yaml_vocab_conversion(temp_path):
	"""Test the json_to_yaml_vocab static method."""
	json_file = Path(temp_path) / "test_vocab.json"
	yaml_file = Path(temp_path) / "test_vocab.yaml"

	dummy_vocab_data = {
		"tokens": ["C", "O", "N"],
		"metadata": {
			"total_tokens": 10,
			"unique_tokens": 3,
			"total_smiles_processed": 3,
			"created_date": "2023-01-01 12:00:00",
			"source": "dummy",
			"charset": "SMILES",
		},
	}
	with open(json_file, "w") as f:
		json.dump(dummy_vocab_data, f)

	SMILESTokenizer.json_to_yaml_vocab(str(json_file), str(yaml_file))

	assert yaml_file.exists()
	with open(yaml_file, "r") as f:
		yaml_content = f.read()
		assert "tokens:" in yaml_content
		assert "- C" in yaml_content
		assert "unique_tokens: 3" in yaml_content


def test_tokenizer_from_json_and_yaml_vocab(temp_path):
	"""Test loading tokenizer from both JSON and YAML vocabulary files."""
	json_file = Path(temp_path) / "test_vocab_load.json"
	yaml_file = Path(temp_path) / "test_vocab_load.yaml"

	dummy_vocab_data = {
		"tokens": ["A", "B", "C"],  # Only A, B, C
		"metadata": {
			"total_tokens": 10,
			"unique_tokens": 3,
			"total_smiles_processed": 3,
			"created_date": "2023-01-01 12:00:00",
			"source": "dummy",
			"charset": "SMILES",
		},
	}
	with open(json_file, "w") as f:
		json.dump(dummy_vocab_data, f)
	SMILESTokenizer.json_to_yaml_vocab(str(json_file), str(yaml_file))  # Ensure YAML exists

	# Load from JSON
	json_tokenizer = SMILESTokenizer.from_vocab_json(str(json_file))

	# The tokenizer will add special tokens and explicit SMILES tokens by default
	# So, the vocab size will be the union of ("A", "B", "C") and default tokens
	# Let's count them: 'A', 'B', 'C' (3) + SMILES_EXPLICIT_TOKENS (121) + SPECIAL_TOKENS (4)
	# Some overlap might exist between SMILES_EXPLICIT_TOKENS and 'A', 'B', 'C' but 'A', 'B', 'C' are not explicitly defined as SMILES_EXPLICIT_TOKENS
	# So the total should be 3 (A, B, C) + 121 (SMILES_EXPLICIT_TOKENS) + 4 (SPECIAL_TOKENS) = 128
	# No, SMILES_EXPLICIT_TOKENS contain 'B', 'C'
	# So it would be: len(set(["A"]) | set(SMILES_EXPLICIT_TOKENS) | set(SPECIAL_TOKENS))
	# Let's verify what the default tokenizer without any input contains
	default_tokenizer_tokens = SMILESTokenizer(None).tokens
	expected_size = len(set(["A", "B", "C"]) | set(default_tokenizer_tokens))  # Union of all unique tokens

	assert json_tokenizer.vocab_size == expected_size
	assert "A" in json_tokenizer.tokens

	yaml_tokenizer = SMILESTokenizer.from_vocab_yaml(str(yaml_file))
	assert yaml_tokenizer.vocab_size == expected_size
	assert "A" in yaml_tokenizer.tokens

	auto_json_tokenizer = SMILESTokenizer.from_vocab(str(json_file))
	assert auto_json_tokenizer.vocab_size == expected_size
	auto_yaml_tokenizer = SMILESTokenizer.from_vocab(str(yaml_file))
	assert auto_yaml_tokenizer.vocab_size == expected_size


def test_vocab_size_getter(full_vocab_tokenizer):
	"""Test the vocab_size getter method."""
	expected_size = len(full_vocab_tokenizer.tokens)  
	assert full_vocab_tokenizer.vocab_size == expected_size
