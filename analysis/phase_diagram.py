#!/usr/bin/env python3
"""
Generate phase diagrams: median tau_grok (log) vs lambda and n.
Produces separate diagrams for each task and batch size.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=str, default="runs/master_results.csv")
    parser.add_argument("--outdir", type=str, default="runs")
    args = parser.parse_args()

    master = pd.read_csv(args.master)
    master = master[master['tau_grok'].notna() & (master['tau_grok'] > 0)]

    for task in master['task'].unique():
        df_task = master[master['task'] == task]
        for batch in df_task['batch_size'].unique():
            subset = df_task[df_task['batch_size'] == batch]
            if subset.empty:
                continue

            pivot = subset.pivot_table(index='n', columns='lambda', values='tau_grok', aggfunc='median')
            pivot_log = np.log10(pivot)

            # Handle single row/column for extent
            lam_min = np.log10(pivot.columns.min())
            lam_max = np.log10(pivot.columns.max())
            n_min = pivot.index.min()
            n_max = pivot.index.max()
            if lam_min == lam_max:
                lam_min, lam_max = lam_min - 0.5, lam_max + 0.5
            if n_min == n_max:
                n_min, n_max = n_min - 1, n_max + 1

            plt.figure(figsize=(8,6))
            im = plt.imshow(pivot_log.values, aspect='auto', cmap='viridis',
                            extent=[lam_min, lam_max, n_min, n_max],
                            origin='lower')
            plt.colorbar(im, label='log10(tau_grok)')
            plt.xlabel('lambda (weight decay)')
            plt.ylabel('n (dataset size)')
            plt.title(f'Phase diagram for {task} (batch={batch})')
            plt.xscale('log')

            # Annotate cells
            for i, n_val in enumerate(pivot.index):
                for j, lam_val in enumerate(pivot.columns):
                    val = pivot_log.iloc[i, j]
                    if not np.isnan(val):
                        plt.text(lam_val, n_val, f'{val:.1f}',
                                 ha='center', va='center', color='white', fontsize=8)
            plt.tight_layout()
            out_file = os.path.join(args.outdir, f'phase_{task}_batch{batch}.png')
            plt.savefig(out_file, dpi=150)
            plt.close()

if __name__ == "__main__":
    main()
