#!/usr/bin/env bash
# reproducibility/reproduce.sh
# Smoke test: verifies that the transformer groks on modular addition.
#
# Expected outcome: runs/smoke_transformer/ contains a log CSV and three
# geometry checkpoints, with test_err reaching 0.000 around step 7000.
#
# Run from the repository root:
#   chmod +x reproducibility/reproduce.sh
#   ./reproducibility/reproduce.sh

set -e

echo "=== Smoke test: modular addition (transformer) ==="
python run_experiment.py \
    --task        modular_add \
    --model       tiny_transformer \
    --n           4000 \
    --batch       512 \
    --wd          0.3 \
    --lr          0.002 \
    --seed        0 \
    --outdir      runs/smoke_transformer \
    --max_steps   20000 \
    --log_interval 25 \
    --grok_threshold 0.1

echo "=== Smoke test complete ==="
