#!/bin/bash

# SMILES Tokenizer - Build and Setup Script
# This script helps you build and test the Rust + PyO3 implementation

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   SMILES Tokenizer - Rust + PyO3 Build and Setup              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check prerequisites
echo "[1/7] Checking prerequisites..."
echo "────────────────────────────────────────────────────────────────"

if ! command -v rustc &> /dev/null; then
    echo "❌ Rust not found. Install from: https://rustup.rs/"
    exit 1
fi
echo "✓ Rust found: $(rustc --version)"

if ! command -v python &> /dev/null; then
    echo "❌ Python not found."
    exit 1
fi
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python found: $python_version"

if ! python -c "import pip" 2>/dev/null; then
    echo "❌ pip not found."
    exit 1
fi
echo "✓ pip found"

# Create virtual environment
echo ""
echo "[2/7] Creating Python virtual environment..."
echo "────────────────────────────────────────────────────────────────"

if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install build tools
echo ""
echo "[3/7] Installing build tools (maturin)..."
echo "────────────────────────────────────────────────────────────────"

pip install --quiet maturin
echo "✓ Maturin installed"

# Build the Rust extension
echo ""
echo "[4/7] Building Rust extension with maturin..."
echo "────────────────────────────────────────────────────────────────"

export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
maturin develop --release
echo "✓ Rust extension built and installed"

# Install test dependencies
echo ""
echo "[5/7] Installing test dependencies..."
echo "────────────────────────────────────────────────────────────────"

pip install --quiet pytest numpy
echo "✓ Test dependencies installed"

# Run tests
echo ""
echo "[6/7] Running basic tests..."
echo "────────────────────────────────────────────────────────────────"

python -c "
from chemformer_rs import SMILESTokenizer, build_vocabulary

# Test 1: Import
print('✓ Module imported successfully')

# Test 2: Build vocabulary
test_smiles = ['CCO', 'CC(O)C', 'c1ccccc1']
vocab, top_tokens = build_vocabulary(test_smiles)
assert len(vocab) > 0, 'Vocabulary is empty'
print(f'✓ Vocabulary built: {len(vocab)} tokens')

# Test 3: Initialize tokenizer
tokenizer = SMILESTokenizer(vocab)
assert tokenizer.vocab_size >= len(vocab), 'Vocab size mismatch'
print(f'✓ Tokenizer initialized: vocab_size={tokenizer.vocab_size}')

# Test 4: Tokenization
tokens, mask = tokenizer.tokenize('CCO')
assert len(tokens) == len(mask), 'Token/mask length mismatch'
print(f'✓ Tokenization works: {len(tokens)} tokens')

# Test 5: Encoding
indices, mask = tokenizer.encode('CCO', add_bos=True, add_eos=True, max_length=10, pad_to_length=True)
assert len(indices) == 10, 'Padding failed'
assert sum(mask) <= 10, 'Mask calculation error'
print(f'✓ Encoding works: {len(indices)} indices, {sum(mask)} valid tokens')

# Test 6: Decoding
decoded = tokenizer.decode(indices)
assert len(decoded) == len(indices), 'Decoding failed'
print(f'✓ Decoding works: {len(decoded)} tokens decoded')

print('')
print('All tests passed! ✓')
"

echo ""
echo "[7/7] Setup complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run example: python example_usage.py"
echo "  3. Run tests: pytest tests/"
echo "  4. Check README.md for API documentation"
echo ""
echo "Quick test:"
echo "  python -c \"from chemformer_rs import SMILESTokenizer; print('✓ Ready to use!')\""
echo ""
