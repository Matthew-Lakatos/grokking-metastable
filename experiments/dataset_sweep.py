#!/usr/bin/env python3
"""
experiments/dataset_sweep.py
Dataset-size sweep for modular addition with a transformer.

Varies training-set size n across [2000, 3000, 4000, 5000, 6000, 7000, 8000]
with 3 seeds each, holding all other hyperparameters fixed at the paper defaults.

Results are saved incrementally to ``runs/dataset_sweep_master.csv`` and a
diagnostic error-bar plot is written to ``runs/dataset_sweep.png``.
The paper-quality figure is produced by ``final_output/analyser.py``.

Run from the repository root:
    python experiments/dataset_sweep.py
"""

import os
import subprocess
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostics.order_params import get_tau_grok

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
TASK         = "modular_add"
MODEL        = "tiny_transformer"
BATCH        = 512
WEIGHT_DECAY = 0.3
LR           = 0.002
DATASET_SIZES = [2000, 3000, 4000, 5000, 6000, 7000, 8000]
SEEDS         = [0, 1, 2]
MAX_STEPS     = 100_000
LOG_INTERVAL  = 25
GROK_THRESHOLD = 0.1

MASTER_CSV = "runs/dataset_sweep_master.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_single(n, seed, outdir):
    """Launch run_experiment.py and return tau_grok."""
    cmd = [
        "python", "run_experiment.py",
        "--task",           TASK,
        "--model",          MODEL,
        "--n",              str(n),
        "--batch",          str(BATCH),
        "--wd",             str(WEIGHT_DECAY),
        "--lr",             str(LR),
        "--seed",           str(seed),
        "--outdir",         outdir,
        "--max_steps",      str(MAX_STEPS),
        "--log_interval",   str(LOG_INTERVAL),
        "--grok_threshold", str(GROK_THRESHOLD),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr.strip()}")
        return np.nan

    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    return get_tau_grok(log_path, grok_threshold=GROK_THRESHOLD)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    os.makedirs("runs", exist_ok=True)

    # Resume.
    results = []
    completed = set()
    if os.path.exists(MASTER_CSV):
        existing = pd.read_csv(MASTER_CSV)
        for _, row in existing.iterrows():
            completed.add((int(row['n']), int(row['seed'])))
        results = existing.to_dict('records')
        print(f"Resuming — {len(completed)} run(s) already completed.")
    else:
        print("Starting new dataset-size sweep.")

    total   = len(DATASET_SIZES) * len(SEEDS)
    run_idx = len(completed)

    for n, seed in product(DATASET_SIZES, SEEDS):
        if (n, seed) in completed:
            continue

        run_idx += 1
        outdir = f"runs/dataset_sweep/n_{n}_seed_{seed}"
        os.makedirs(outdir, exist_ok=True)

        print(f"[{run_idx}/{total}] n={n}  seed={seed}")
        t0 = time.time()
        tau = run_single(n, seed, outdir)
        elapsed = time.time() - t0

        if np.isnan(tau):
            print(f"  -> No grokking detected  ({elapsed / 60:.1f} min)")
        else:
            print(f"  -> tau_grok={tau:.0f}  ({elapsed / 60:.1f} min)")

        results.append({'n': n, 'seed': seed, 'tau_grok': tau})
        pd.DataFrame(results).to_csv(MASTER_CSV, index=False)

    print(f"\nAll runs complete. Results saved to {MASTER_CSV!r}.")


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def plot_summary():
    """
    Error-bar plot of median grokking time vs. dataset size.

    Diagnostic only; authoritative figure comes from final_output/analyser.py.
    """
    df = pd.read_csv(MASTER_CSV)
    df = df[df['tau_grok'].notna()]

    if df.empty:
        print("No valid grokking times to plot.")
        return

    summary = df.groupby('n')['tau_grok'].agg(
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).reset_index()
    summary['yerr_low']  = summary['median'] - summary['q25']
    summary['yerr_high'] = summary['q75']   - summary['median']

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary['n'], summary['median'],
        yerr=[summary['yerr_low'], summary['yerr_high']],
        fmt='o-', capsize=5,
    )
    plt.xlabel('Dataset size n')
    plt.ylabel('Grokking time τ (steps)')
    plt.title('Dataset-size sweep — modular addition (transformer) [diagnostic]')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = "runs/dataset_sweep.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Diagnostic plot saved to {out_path!r}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_sweep()
    plot_summary()
