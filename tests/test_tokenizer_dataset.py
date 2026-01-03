import pytest
import os
from pathlib import Path
from omegaconf import OmegaConf

from chemformer_rs.tokenizer import SMILESTokenizer
from dataset.zinc import ZincDataset
from dataset.uspto_sep import UsptoSepDataset

# --- Configuration for tests ---
# Paths are relative to the project root for these tests
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config"
ZINC_LMDB_PATH = PROJECT_ROOT / "data" / "zinc.lmdb"
USPTO_LMDB_PATH = PROJECT_ROOT / "data" / "uspto_sep.lmdb"
VOCAB_PATH = CONFIG_PATH / "vocab.yaml"
MAX_LENGTH = 128 # A reasonable default for testing, could be read from config if needed


@pytest.fixture(scope="module")
def tokenizer():
    """Loads the main tokenizer for all tests in this module."""
    # This might need to be adapted if the tokenizer adds '>' dynamically
    tokenizer = SMILESTokenizer.from_vocab(str(VOCAB_PATH))
    if ">" not in tokenizer.tokens:
        tokenizer.add_token(">")
    return tokenizer

def check_for_unk_tokens(tokenizer_instance, dataset_name, dataset_instance, num_samples=1000):
    """
    Helper function to check for UNK tokens in a dataset.
    num_samples: Check only this many samples to keep test fast.
    """
    unk_token_id = tokenizer_instance.token_to_index("<UNK>")
    found_unk_smiles = []

    if len(dataset_instance) == 0:
        pytest.skip(f"Dataset '{dataset_name}' is empty, skipping UNK check.")

    for i in range(min(num_samples, len(dataset_instance))):
        sample = dataset_instance[i]
        
        # Check source and target sequences for UNK tokens
        for key in ["src", "tgt"]:
            token_ids = sample[key].numpy()
            if unk_token_id in token_ids:
                # Attempt to decode the original SMILES for better debugging
                if key == "src":
                    # For ZINC, src is masked, so we need original SMILES from tgt
                    if dataset_name == "ZINC": # Use string name for comparison
                        original_smiles_tokens = tokenizer_instance.decode(sample["tgt"].numpy())
                    else: # USPTO, src is reactants, reagents
                        # Decode the source tokens if UNK is found there
                        original_smiles_tokens = tokenizer_instance.decode(sample["src"].numpy())
                elif key == "tgt":
                    original_smiles_tokens = tokenizer_instance.decode(sample["tgt"].numpy())
                
                clean_smiles = "".join([t for t in original_smiles_tokens if t not in ["<PAD>", "<BOS>", "<EOS>", "<MASK>"]]) # Add <MASK> for ZINC
                
                found_unk_smiles.append(f"UNK in {dataset_name} ({key}) - Index {i}: Decoded: {''.join(tokenizer_instance.decode(token_ids))}, Original: {clean_smiles}")
                
                # Only report once per sample
                break 

    assert not found_unk_smiles, (
        f"Found <UNK> tokens in {len(found_unk_smiles)} samples from {dataset_name} dataset:\n"
        + "\n".join(found_unk_smiles)
    )

@pytest.mark.skipif(not ZINC_LMDB_PATH.exists(), reason=f"ZINC LMDB not found at {ZINC_LMDB_PATH}")
def test_zinc_dataset_for_unk_tokens(tokenizer):
    """Checks the ZINC dataset (test split) for unknown tokens."""
    cfg_zinc = OmegaConf.load(CONFIG_PATH / "task" / "pretrain_zinc.yaml")
    
    zinc_ds = ZincDataset(
        lmdb_path=str(ZINC_LMDB_PATH),
        subset_indices=ZincDataset.read_split_indices(str(ZINC_LMDB_PATH), "test"),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        is_training=False, # For testing, canonical SMILES
        # Default masking args from config, should be fine for just checking tokens
        mask_prob=cfg_zinc.dataset.get("mask_prob", 0.3),
        span_len=cfg_zinc.dataset.get("span_len", 3),
        augment_prob=0.0, # No augmentation for testing
        span_mask_proportion=cfg_zinc.dataset.get("span_mask_proportion", 1.0),
        span_random_proportion=cfg_zinc.dataset.get("span_random_proportion", 0.0),
    )
    check_for_unk_tokens(tokenizer, "ZINC", zinc_ds)

@pytest.mark.skipif(not USPTO_LMDB_PATH.exists(), reason=f"USPTO LMDB not found at {USPTO_LMDB_PATH}")
def test_uspto_dataset_for_unk_tokens(tokenizer):
    """Checks the USPTO dataset (test split) for unknown tokens."""
    # USPTO does not have masking args
    uspto_ds = UsptoSepDataset(
        lmdb_path=str(USPTO_LMDB_PATH),
        subset_indices=UsptoSepDataset.read_split_indices(str(USPTO_LMDB_PATH), "test"),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        is_training=False, # For testing, canonical SMILES
        augment_prob=0.0, # No augmentation for testing
    )
    check_for_unk_tokens(tokenizer, "USPTO", uspto_ds)
