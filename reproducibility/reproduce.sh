#!/usr/bin/env bash
set -e  # stop on first error

echo "=== Smoke test: modular addition ==="
python run_experiment.py \
    --task modular_add \
    --model tiny_mlp \
    --n 500 \
    --batch 64 \
    --wd 1e-5 \
    --seed 0 \
    --outdir runs/smoke_modular \
    --max_steps 2000 \
    --log_interval 100 \
    --grok_threshold 0.1

echo "=== Smoke test: sparse parity ==="
python run_experiment.py \
    --task sparse_parity \
    --model tiny_mlp \
    --n 500 \
    --batch 64 \
    --wd 1e-2 \
    --seed 0 \
    --outdir runs/smoke_sparse \
    --max_steps 2000 \
    --log_interval 100 \
    --grok_threshold 0.1

echo "Both smoke tests completed successfully."
