#!/usr/bin/env python3
"""
analysis/phase_diagram.py
Generate phase diagrams: median τ_grok (log₁₀) as a heatmap over λ and n.

Reads the master results CSV and produces one figure per (task, batch_size)
combination.

Usage:
    python analysis/phase_diagram.py
    python analysis/phase_diagram.py --master runs/master_results.csv --outdir runs
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_phase_diagram(df_task, task, batch, outdir):
    """Render and save a single phase-diagram heatmap."""
    subset = df_task[df_task['batch_size'] == batch]
    if subset.empty:
        return

    pivot     = subset.pivot_table(index='n', columns='lambda',
                                   values='tau_grok', aggfunc='median')
    pivot_log = np.log10(pivot)

    lam_min = np.log10(pivot.columns.min())
    lam_max = np.log10(pivot.columns.max())
    n_min   = pivot.index.min()
    n_max   = pivot.index.max()

    # Avoid degenerate extents.
    if lam_min == lam_max:
        lam_min, lam_max = lam_min - 0.5, lam_max + 0.5
    if n_min == n_max:
        n_min, n_max = n_min - 1, n_max + 1

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        pivot_log.values, aspect='auto', cmap='viridis',
        extent=[lam_min, lam_max, n_min, n_max], origin='lower',
    )
    plt.colorbar(im, label='log₁₀(τ_grok)')
    plt.xlabel('λ (weight decay)')
    plt.ylabel('n (dataset size)')
    plt.title(f'Phase diagram — {task}  (batch={batch})')
    plt.xscale('log')

    # Annotate each cell.
    for i, n_val in enumerate(pivot.index):
        for j, lam_val in enumerate(pivot.columns):
            val = pivot_log.iloc[i, j]
            if not np.isnan(val):
                plt.text(lam_val, n_val, f'{val:.1f}',
                         ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(outdir, f'phase_{task}_batch{batch}.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Phase diagram saved to {out_path!r}.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate phase diagrams of median τ_grok vs λ and n."
    )
    parser.add_argument("--master", type=str, default="runs/master_results.csv",
                        help="Path to the master results CSV.")
    parser.add_argument("--outdir", type=str, default="runs",
                        help="Output directory for figures.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    master = pd.read_csv(args.master)
    master = master[master['tau_grok'].notna() & (master['tau_grok'] > 0)]

    for task in master['task'].unique():
        df_task = master[master['task'] == task]
        for batch in df_task['batch_size'].unique():
            plot_phase_diagram(df_task, task, batch, args.outdir)


if __name__ == "__main__":
    main()
