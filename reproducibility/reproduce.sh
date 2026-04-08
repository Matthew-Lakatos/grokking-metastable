#!/usr/bin/env bash
python run_experiment.py --config configs/modular_addition.yaml
python analysis/plot_trajectories.py --log runs/smoke/log_seed0.csv --out runs/smoke/trajectory.png
