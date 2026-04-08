#!/usr/bin/env python3
"""
Generate phase diagrams: median tau_grok (log) vs lambda and n, for each batch size.
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
    # Filter only runs with valid tau_grok
    master = master[master['tau_grok'].notna() & (master['tau_grok'] > 0)]
    
    for batch in [8, 64]:
        subset = master[master['batch_size'] == batch]
        if subset.empty:
            continue
        # Pivot table: median tau_grok (log) over seeds
        pivot = subset.pivot_table(index='n', columns='lambda', values='tau_grok', aggfunc='median')
        # Log transform
        pivot_log = np.log10(pivot)
        # Plot heatmap
        plt.figure(figsize=(8,6))
        im = plt.imshow(pivot_log.values, aspect='auto', cmap='viridis', 
                        extent=[np.log10(pivot.columns.min()), np.log10(pivot.columns.max()),
                                pivot.index.min(), pivot.index.max()],
                        origin='lower')
        plt.colorbar(im, label='log10(tau_grok)')
        plt.xlabel('lambda (weight decay)')
        plt.ylabel('n (dataset size)')
        plt.title(f'Phase diagram: median grokking time (batch={batch})')
        plt.xscale('log')
        # Add text annotations
        for i, n in enumerate(pivot.index):
            for j, lam in enumerate(pivot.columns):
                val = pivot_log.iloc[i, j]
                if not np.isnan(val):
                    plt.text(lam, n, f'{val:.1f}', ha='center', va='center', color='white', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f'phase_lambda_n_batch{batch}.png'), dpi=150)
        plt.close()

if __name__ == "__main__":
    main()
