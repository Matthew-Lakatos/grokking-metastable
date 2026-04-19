#!/usr/bin/env python3
"""
final_output/analyser.py
Authoritative analysis script — produces all paper figures and result CSVs.

Discovers all ``log_seed*.csv`` files under ``runs/`` by directory-name
convention, applies the corrected grokking-detection criteria, and generates:

  final_output/all_sweeps_reanalysed.csv  – Combined results for all sweeps.
  final_output/arrhenius_corrected.png    – Arrhenius plot (log τ vs B/lr).
  final_output/dataset_corrected.png      – Dataset-size sweep error-bar plot.
  final_output/lambda_corrected.png       – Lambda sweep error-bar plot.
  final_output/{sweep}_corrected.csv      – Per-sweep results.

Grokking-detection criteria (authoritative):
  - test_err < 0.1
  - test_err stays below 0.1 for 5 consecutive log entries
  - train_loss drops below 0.5 at some point in the log

Temperature proxy for the Arrhenius plot:
  1/T_eff ≈ B / lr   (batch size / learning rate)
  This is distinct from the FlucDis T_eff_proxy column in the log.

Usage:
    python final_output/analyser.py
    python final_output/analyser.py --base_dir . --outdir final_output

Run from the repository root.
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from diagnostics.order_params import get_tau_grok

# Batch size is fixed at 512 across all paper experiments.
BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Log-file discovery and parsing
# ---------------------------------------------------------------------------

def discover_logs(base_dir):
    """
    Find all log_seed*.csv files under base_dir/runs/ and parse sweep type
    and parameter from their parent directory name.

    Returns a list of dicts with keys: path, sweep, param, seed.
    """
    pattern = os.path.join(base_dir, "runs", "**", "log_seed*.csv")
    log_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(log_files)} log file(s).")

    records = []
    for path in log_files:
        parts = path.split(os.sep)
        dir_name = parts[-2]

        if 'dataset_sweep' in path:
            m = re.search(r'n_(\d+)_seed_(\d+)', dir_name)
            if not m:
                continue
            sweep = 'dataset'
            param = int(m.group(1))
            seed  = int(m.group(2))

        elif 'lambda_sweep' in path:
            m = re.search(r'wd_(\d+\.?\d*)_seed_(\d+)', dir_name)
            if not m:
                continue
            sweep = 'lambda'
            param = float(m.group(1))
            seed  = int(m.group(2))

        elif 'arrhenius' in path:
            m = re.search(r'lr_(\d+\.?\d*)_seed_(\d+)', dir_name)
            if not m:
                continue
            sweep = 'arrhenius'
            param = float(m.group(1))
            seed  = int(m.group(2))

        else:
            continue

        records.append(dict(path=path, sweep=sweep, param=param, seed=seed))

    return records


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def plot_arrhenius(subset, outdir):
    """Scatter + linear fit of log τ vs B/lr."""
    subset = subset[subset['tau_grok'].notna()].copy()
    if subset.empty:
        print("  No valid grokking times for Arrhenius sweep — skipping.")
        return

    subset['inv_T']   = BATCH_SIZE / subset['param']   # B / lr
    subset['log_tau'] = np.log(subset['tau_grok'])

    slope, intercept, r_value, p_value, _ = stats.linregress(
        subset['inv_T'], subset['log_tau']
    )
    print(f"\n  Arrhenius fit:  log(τ) = {intercept:.4f} + {slope:.6f} × (B/lr)"
          f"\n  R²={r_value**2:.4f}  p={p_value:.2e}")

    plt.figure(figsize=(6, 5))
    plt.scatter(subset['inv_T'], subset['log_tau'], alpha=0.6, label='Data')
    x_line = np.linspace(subset['inv_T'].min(), subset['inv_T'].max(), 100)
    plt.plot(x_line, intercept + slope * x_line, 'r-',
             label=f'fit: slope={slope:.6f}')
    plt.xlabel('B / lr  (batch size / learning rate)')
    plt.ylabel('log(τ_grok)')
    plt.title('Arrhenius scaling — modular addition (transformer)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(outdir, 'arrhenius_corrected.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path!r}.")


def plot_errorbar(subset, sweep, outdir):
    """Error-bar plot of median τ ± IQR vs the sweep parameter."""
    summary = subset.groupby('param')['tau_grok'].agg(
        median='median',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).reset_index()
    valid = summary[summary['median'].notna()].copy()
    if valid.empty:
        print(f"  No valid grokking times for {sweep} sweep — skipping.")
        return
    valid['yerr_low']  = valid['median'] - valid['q25']
    valid['yerr_high'] = valid['q75']   - valid['median']

    xlabel = 'Dataset size n' if sweep == 'dataset' else 'Weight decay λ'
    plt.figure(figsize=(8, 5))
    plt.errorbar(valid['param'], valid['median'],
                 yerr=[valid['yerr_low'], valid['yerr_high']],
                 fmt='o-', capsize=5)
    plt.xlabel(xlabel)
    plt.ylabel('Grokking time τ (steps)')
    plt.title(f'{sweep.capitalize()} sweep')
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(outdir, f'{sweep}_corrected.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved {out_path!r}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Produce authoritative paper figures from all sweep logs."
    )
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Repository root (parent of runs/).")
    parser.add_argument("--outdir",   type=str, default="final_output",
                        help="Output directory for figures and CSVs.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    records = discover_logs(args.base_dir)
    if not records:
        print("No log files found. Check --base_dir and your runs/ folder.")
        return

    # Detect grokking for every log.
    data = []
    for rec in records:
        tau = get_tau_grok(rec['path'])
        print(f"  {rec['sweep']}  param={rec['param']}  seed={rec['seed']}"
              f"  tau={tau}")
        data.append({
            'sweep': rec['sweep'],
            'param': rec['param'],
            'seed':  rec['seed'],
            'tau_grok': tau,
        })

    df = pd.DataFrame(data)
    combined_path = os.path.join(args.outdir, 'all_sweeps_reanalysed.csv')
    df.to_csv(combined_path, index=False)
    print(f"\nCombined results saved to {combined_path!r}.")

    # Generate one figure and one CSV per sweep type.
    for sweep in df['sweep'].unique():
        subset = df[df['sweep'] == sweep].copy()
        print(f"\n=== {sweep.upper()} sweep ===")

        if sweep == 'arrhenius':
            plot_arrhenius(subset, args.outdir)
        else:
            plot_errorbar(subset, sweep, args.outdir)

        per_sweep_path = os.path.join(args.outdir, f'{sweep}_corrected.csv')
        subset.to_csv(per_sweep_path, index=False)
        print(f"  Per-sweep CSV saved to {per_sweep_path!r}.")


if __name__ == "__main__":
    main()
