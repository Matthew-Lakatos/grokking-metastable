#!/usr/bin/env python3
"""
Fit Arrhenius scaling: log(tau_grok) ~ a + b / T_eff_proxy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import argparse

def extract_T_eff_proxy(csv_path, tau_grok_step):
    """Extract T_eff_proxy from run log, using median of last 100 steps before grokking."""
    if not os.path.exists(csv_path):
        return np.nan
    df = pd.read_csv(csv_path)
    if 'T_eff_proxy' not in df.columns:
        return np.nan
    # Find steps before grokking (step <= tau_grok_step)
    df_before = df[df['step'] <= tau_grok_step]
    if len(df_before) == 0:
        return np.nan
    # Use median of last up to 100 steps
    n = min(100, len(df_before))
    T_vals = df_before['T_eff_proxy'].iloc[-n:].values
    T_vals = T_vals[~np.isnan(T_vals)]
    if len(T_vals) == 0:
        return np.nan
    return np.median(T_vals)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=str, default="runs/master_results.csv")
    parser.add_argument("--outdir", type=str, default="runs")
    args = parser.parse_args()
    
    master = pd.read_csv(args.master)
    # Filter runs that grokked (tau_grok finite and > 0)
    master = master[(master['tau_grok'].notna()) & (master['tau_grok'] > 0)]
    
    data = []
    for idx, row in master.iterrows():
        run_dir = f"runs/lambda_{row['lambda']}_n_{row['n']}_batch_{row['batch_size']}_seed_{row['seed']}"
        csv_path = os.path.join(run_dir, f"log_seed{row['seed']}.csv")
        T_eff = extract_T_eff_proxy(csv_path, row['tau_grok'])
        if not np.isnan(T_eff) and T_eff > 0:
            data.append({
                'tau_grok': row['tau_grok'],
                'T_eff': T_eff,
                'lambda': row['lambda'],
                'n': row['n']
            })
    
    if len(data) < 3:
        print("Not enough valid runs for Arrhenius fit.")
        return
    
    df_fit = pd.DataFrame(data)
    df_fit['log_tau'] = np.log(df_fit['tau_grok'])
    df_fit['inv_T'] = 1.0 / df_fit['T_eff']
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_fit['inv_T'], df_fit['log_tau'])
    
    print(f"Arrhenius fit: log(tau) = {intercept:.3f} + {slope:.3f} * (1/T_eff)")
    print(f"R² = {r_value**2:.4f}, p = {p_value:.2e}")
    
    # Plot
    plt.figure(figsize=(6,5))
    plt.scatter(df_fit['inv_T'], df_fit['log_tau'], alpha=0.6, label='Data')
    x_line = np.linspace(df_fit['inv_T'].min(), df_fit['inv_T'].max(), 100)
    plt.plot(x_line, intercept + slope * x_line, 'r-', label=f'fit: slope={slope:.3f}')
    plt.xlabel('1 / T_eff_proxy')
    plt.ylabel('log(tau_grok)')
    plt.title('Arrhenius scaling for grokking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'arrhenius_fit.png'), dpi=150)
    plt.show()
    
    # Save fit summary
    with open(os.path.join(args.outdir, 'arrhenius_summary.txt'), 'w') as f:
        f.write(f"slope = {slope:.6f}\n")
        f.write(f"intercept = {intercept:.6f}\n")
        f.write(f"R_squared = {r_value**2:.6f}\n")
        f.write(f"p_value = {p_value:.6e}\n")
        f.write(f"std_err = {std_err:.6f}\n")

if __name__ == "__main__":
    main()
