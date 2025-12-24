from chemformer_rs.tokenizer import SMILESTokenizer


def main():
	"""
	Converts the vocabulary from JSON to YAML format.
	"""
	json_path = "rocksdb_databases/vocab_unified_all_datasets/vocab_metadata.json"
	yaml_path = "config/vocab.yaml"
	SMILESTokenizer.json_to_yaml_vocab(json_path, yaml_path)
	print(f"Successfully converted {json_path} to {yaml_path}")


if __name__ == "__main__":
	main()
