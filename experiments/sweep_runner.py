#!/usr/bin/env python3

# need python 3.7+ due to subprocess.run and capture_output=True
# for older versions, remove capture_output and use stdout=PIPE, stderr=PIPE

"""
Sweep runner for grokking experiments
Runs both modular addition and sparse parity with:
- λ = [1e-3, 1e-2, 1e-1]
- n = [500, 1000]
- batch_size = 64
- seeds = 0,1,2,3,4
- max_steps = 50000 (n=500) or 100000 (n=1000)
- grok_threshold = 0.1
- Logs tau_grok and T_eff_proxy, saves master_results.csv.
"""

import subprocess
import pandas as pd
import numpy as np
import os
import time
from itertools import product

# helper: computes tau_grok and T_eff_proxy from a CSV log from a run
def get_tau_grok_and_teff(csv_path, grok_threshold=0.1, train_loss_thresh=1e-3, min_residence=100):
    """
    Returns (tau_grok, T_eff_proxy_at_grok) or (nan, nan) if conditions not met.
    Conditions:
      - test_err first drops below grok_threshold
      - train_loss was below train_loss_thresh for at least min_residence steps before that
    """
    if not os.path.exists(csv_path):
        return np.nan, np.nan
    df = pd.read_csv(csv_path)
    if 'T_eff_proxy' not in df.columns or 'train_loss' not in df.columns:
        return np.nan, np.nan

    grok_mask = df['test_err'] < grok_threshold
    if not grok_mask.any():
        return np.nan, np.nan
    first_grok = df[grok_mask]['step'].iloc[0]

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
        teff_row = df_before.iloc[-1:]
    teff_val = teff_row['T_eff_proxy'].values[0]
    return first_grok, teff_val

# main sweep
def main():
    # Experiment grid
    tasks = ["modular_add", "sparse_parity"]
    lambdas = [1e-3, 1e-2, 1e-1]          # weight decay (regularisation)
    ns = [500, 1000]                       # training set sizes
    batch_size = 64                        # fixed batch size
    seeds = [0, 1, 2, 3, 4]               # 5 seeds for statistics
    grok_threshold = 0.1

    # Steps: 50k for n=500, 100k for n=1000
    max_steps_dict = {500: 50000, 1000: 100000}

    # Output file
    os.makedirs("runs", exist_ok=True)
    master_path = "runs/master_results.csv"

    # Load already completed runs
    results = []
    completed = set()
    if os.path.exists(master_path):
        existing = pd.read_csv(master_path)
        for _, row in existing.iterrows():
            key = (row['task'], row['lambda'], row['n'], row['batch_size'], row['seed'])
            completed.add(key)
        results = existing.to_dict('records')
        print(f"Resuming: {len(completed)} runs already completed.")
    else:
        print("Starting new sweep.")

    total_runs = len(tasks) * len(lambdas) * len(ns) * len(seeds)
    run_idx = len(completed)
    start_time = time.time()

    for task, wd, n, seed in product(tasks, lambdas, ns, seeds):
        key = (task, wd, n, batch_size, seed)
        if key in completed:
            continue

        run_idx += 1
        max_steps = max_steps_dict[n]
        outdir = f"runs/{task}/lambda_{wd}_n_{n}_batch_{batch_size}_seed_{seed}"
        os.makedirs(outdir, exist_ok=True)

        cmd = [
            "python", "run_experiment.py",
            "--task", task,
            "--model", "tiny_mlp",
            "--n", str(n),
            "--batch", str(batch_size),
            "--wd", str(wd),
            "--seed", str(seed),
            "--outdir", outdir,
            "--max_steps", str(max_steps),
            "--log_interval", "200",
            "--grok_threshold", str(grok_threshold)
        ]

        print(f"[{run_idx}/{total_runs}] Running: {task} | λ={wd} | n={n} | seed={seed} (steps={max_steps})")
        start_run = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_run = time.time() - start_run

        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            tau_grok, teff = np.nan, np.nan
        else:
            csv_path = os.path.join(outdir, f"log_seed{seed}.csv")
            tau_grok, teff = get_tau_grok_and_teff(csv_path, grok_threshold)
            print(f"  Completed in {elapsed_run:.1f}s, tau_grok={tau_grok}, T_eff={teff:.3e}")

        results.append({
            'task': task,
            'lambda': wd,
            'n': n,
            'batch_size': batch_size,
            'seed': seed,
            'tau_grok': tau_grok,
            'T_eff_proxy': teff
        })

        # Save incrementally
        pd.DataFrame(results).to_csv(master_path, index=False)

        elapsed_total = time.time() - start_time
        remaining = (total_runs - run_idx) * (elapsed_total / run_idx) if run_idx > 0 else 0
        print(f"  Total elapsed: {elapsed_total/3600:.2f}h, estimated remaining: {remaining/3600:.2f}h")

    print("\nAll experiments completed. Results saved to", master_path)

if __name__ == "__main__":
    main()
