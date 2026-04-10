#!/usr/bin/env bash
set -e

echo "=== Smoke test: modular addition (transformer) ==="
python run_experiment.py \
    --task modular_add \
    --model tiny_transformer \
    --n 4000 \
    --batch 512 \
    --wd 0.3 \
    --lr 0.002 \
    --seed 0 \
    --outdir runs/smoke_transformer \
    --max_steps 20000 \
    --log_interval 1000 \
    --grok_threshold 0.1

echo "Both smoke tests completed successfully."
