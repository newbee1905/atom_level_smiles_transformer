#!/bin/bash

# This script provides a convenient way to launch local training using torchrun.
# It automatically detects the number of available NVIDIA GPUs and launches one
# training process per GPU. If no GPUs are found, it defaults to a single
# process, suitable for CPU training.
#
# Usage:
#   ./train_local.sh [hydra_options]
#
# Examples:
#   # Train locally on all available GPUs
#   ./train_local.sh
#
#   # Override the batch size
#   ./train_local.sh training.batch_size=32
#

set -euo pipefail

# Activate the virtual environment
if [ -d ".venv" ]; then
	echo "Activating virtual environment from ./.venv"
	source .venv/bin/activate
else
	echo "Warning: .venv directory not found. Assuming environment is already active."
fi

if [ -z "${PYTHONPATH:-}" ]; then
	export PYTHONPATH=.
else
	export PYTHONPATH="$PYTHONPATH:."
fi

if command -v nvidia-smi &> /dev/null; then
	NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader)
else
	echo "nvidia-smi not found, defaulting to 1 process."
	NUM_GPUS=1
fi

echo "Found $NUM_GPUS device(s). Starting local training with torchrun..."
echo "Any additional arguments will be passed to main.py: $@"

torchrun \
	--nproc_per_node=$NUM_GPUS \
	main.py "$@"

echo "Local training complete."
