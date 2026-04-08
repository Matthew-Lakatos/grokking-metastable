#!/usr/bin/env python3
"""
Sweep runner for grokking experiments.
Runs grid over weight_decay (lambda), dataset size (n), batch_size, seed.
Collects tau_grok (first step where test_err < threshold and training loss was low for a while)
and T_eff_proxy at grokking.
"""

import subprocess
import pandas as pd
import numpy as np
import os
import time
from itertools import product

def get_tau_grok_and_teff(csv_path, grok_threshold=0.1, train_loss_thresh=1e-3, min_residence=100):
    """
    Returns (tau_grok, T_eff_proxy_at_grok) or (nan, nan) if conditions not met.
    T_eff_proxy is taken from the row where test_err first drops below threshold.
    """
    if not os.path.exists(csv_path):
        return np.nan, np.nan
    df = pd.read_csv(csv_path)
    if 'T_eff_proxy' not in df.columns:
        return np.nan, np.nan
    # Find first grok step
    grok_mask = df['test_err'] < grok_threshold
    if not grok_mask.any():
        return np.nan, np.nan
    first_grok = df[grok_mask]['step'].iloc[0]
    # Check train loss before grok
    df_before = df[df['step'] <= first_grok]
    low_loss_mask = df_before['train_loss'] < train_loss_thresh
    if not low_loss_mask.any():
        return np.nan, np.nan
    t0 = df_before[low_loss_mask]['step'].iloc[0]
    if first_grok - t0 < min_residence:
        return np.nan, np.nan
    # Get T_eff_proxy at grokking step (or nearest)
    teff_row = df[df['step'] == first_grok]
    if teff_row.empty:
        # fallback: use last available before grok
        teff_row = df_before.iloc[-1:]
    teff_val = teff_row['T_eff_proxy'].values[0]
    return first_grok, teff_val

def main():
    # Define grid
    lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
    ns = [100, 500, 2000]
    batch_sizes = [8, 64]
    seeds = [0, 1, 2]
    grok_threshold = 0.1  # consistent with run_experiment default

    results = []
    os.makedirs("runs", exist_ok=True)

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
            "--max_steps", "20000",
            "--grok_threshold", str(grok_threshold)
        ]
        print(f"Running: {' '.join(cmd)}")
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time
        if result.returncode != 0:
            print(f"Error in run: {result.stderr}")
            tau_grok = np.nan
            teff = np.nan
        else:
            csv_path = os.path.join(outdir, f"log_seed{seed}.csv")
            tau_grok, teff = get_tau_grok_and_teff(csv_path, grok_threshold=grok_threshold)
            print(f"Completed in {elapsed:.1f}s, tau_grok={tau_grok}, T_eff={teff:.3e}")
        
        results.append({
            'lambda': wd,
            'n': n,
            'batch_size': bs,
            'seed': seed,
            'tau_grok': tau_grok,
            'T_eff_proxy': teff
        })
        
        # Save incrementally
        df_res = pd.DataFrame(results)
        df_res.to_csv("runs/master_results.csv", index=False)

    print("Sweep finished. Results in runs/master_results.csv")

if __name__ == "__main__":
    main()
