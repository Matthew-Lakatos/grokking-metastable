#!/usr/bin/env bash
python run_experiment.py \
    --task modular_add \
    --model tiny_mlp \
    --n 500 \
    --batch 8 \
    --wd 1e-5 \
    --seed 0 \
    --outdir runs/smoke \
    --max_steps 2000 \
    --log_interval 100
