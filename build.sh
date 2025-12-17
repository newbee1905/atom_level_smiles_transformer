#!/bin/bash

source .venv/bin/activate

maturin develop --release --manifest-path chemformer_rs/Cargo.toml
