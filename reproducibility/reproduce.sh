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

echo "Smoke tests completed successfully."
