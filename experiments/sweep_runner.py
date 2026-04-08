#!/usr/bin/env python3
"""
Sweep runner for grokking experiments.
Usage: python experiments/sweep_runner.py
"""

import subprocess
import pandas as pd
import numpy as np
import os
import time
from itertools import product

# Define grid
lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
ns = [100, 500, 2000]
batch_sizes = [8, 64]
seeds = [0, 1, 2]

results = []

# Function to extract tau_grok from a CSV log
def get_tau_grok(csv_path, grok_threshold=0.1, train_loss_thresh=1e-3, min_residence=100):
    """
    Returns first step where test_err < grok_threshold AND 
    train_loss was below train_loss_thresh for at least min_residence steps before.
    If not satisfied, returns nan.
    """
    if not os.path.exists(csv_path):
        return np.nan
    df = pd.read_csv(csv_path)
    # Find first grok step
    grok_steps = df[df['test_err'] < grok_threshold]['step']
    if grok_steps.empty:
        return np.nan
    first_grok = grok_steps.iloc[0]
    # Check train loss before grok
    df_before = df[df['step'] <= first_grok]
    # Find step where train_loss first goes below threshold
    low_loss_steps = df_before[df_before['train_loss'] < train_loss_thresh]['step']
    if low_loss_steps.empty:
        return np.nan
    t0 = low_loss_steps.iloc[0]
    if first_grok - t0 >= min_residence:
        return first_grok
    else:
        return np.nan

# Run each config
for wd, n, bs, seed in product(lambdas, ns, batch_sizes, seeds):
    outdir = f"runs/lambda_{wd}_n_{n}_batch_{bs}_seed_{seed}"
    cmd = [
        "python", "run_experiment.py",
        "--task", "modular_add",
        "--model", "tiny_mlp",
        "--n", str(n),
        "--batch", str(bs),
        "--wd", str(wd),
        "--seed", str(seed),
        "--outdir", outdir,
        "--max_steps", "20000"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)  # check=False to continue on error
    
    csv_path = os.path.join(outdir, f"log_seed{seed}.csv")
    tau = get_tau_grok(csv_path)
    results.append({
        'lambda': wd,
        'n': n,
        'batch_size': bs,
        'seed': seed,
        'tau_grok': tau
    })
    
    # Save incremental results
    df_res = pd.DataFrame(results)
    df_res.to_csv("runs/master_results.csv", index=False)

print("Sweep finished. Results in runs/master_results.csv")
