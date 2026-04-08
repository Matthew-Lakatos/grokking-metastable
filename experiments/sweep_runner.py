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
def get_tau_grok(csv_path):
    if not os.path.exists(csv_path):
        return np.nan
    df = pd.read_csv(csv_path)
    # Find first step where test_err < 0.1
    mask = df['test_err'] < 0.1
    if mask.any():
        return df.loc[mask.idxmax(), 'step']
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
