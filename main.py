import hydra
from omegaconf import DictConfig, OmegaConf
from model.bart import Bart
from chemformer_rs.tokenizer import SMILESTokenizer


def count_parameters(model):
	"""
	Counts the total number of trainable parameters in a given PyTorch model.
	"""
	num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	if num_params >= 1_000_000:
		return f"{num_params / 1_000_000:.1f}M"
	elif num_params >= 1_000:
		return f"{num_params / 1_000:.1f}K"
	else:
		return str(num_params)


@hydra.main(config_path="config", config_name="model", version_base="1.3")
def main(cfg: DictConfig):
	"""
	Initializes a BART model using Hydra configuration and prints its total number of parameters.
	"""
	print(f"Hydra configuration:\n{OmegaConf.to_yaml(cfg)}")

	tokenizer = SMILESTokenizer.from_vocab("config/vocab.yaml")
	print(f"Tokenizer loaded with {tokenizer.vocab_size} tokens.")

	test_smiles = "CCOc1ccc(C=C(C#N)C(=O)O)cc1"
	encoded, _ = tokenizer.encode(test_smiles)
	decoded = tokenizer.decode_to_string(encoded)
	print(f"Original SMILES: {test_smiles}")
	print(f"Encoded token ids: {encoded}")
	print(f"Decoded SMILES: {decoded}")

	# Temporarily disable strict mode
	OmegaConf.set_struct(cfg, False)  
	cfg.vocab_size = tokenizer.vocab_size
	OmegaConf.set_struct(cfg, True) 

	model = Bart(cfg)

	num_params = count_parameters(model)
	print(f"Total number of trainable parameters in the BART model: {num_params}")


if __name__ == "__main__":
	main()
