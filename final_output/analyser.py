# %% [markdown]
# # analyse All Sweeps with Corrected Grokking Detection
# 
# - train_loss_thresh = 0.5
# - min_stable = 5 (test error must stay below 0.1 for 5 consecutive logs)
# - No requirement that train loss drops before test error
# - Arrhenius sweep: scatter plot + linear regression (log τ vs B/η)
# - Lambda sweep: errorbar plot (median ± IQR)

# %% [code]
import os, glob, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def get_tau_grok_corrected(log_path, grok_threshold=0.1, train_loss_thresh=0.5, min_stable=5):
    if not os.path.exists(log_path):
        return np.nan
    df = pd.read_csv(log_path)
    if 'test_err' not in df.columns or 'train_loss' not in df.columns:
        return np.nan
    grok_mask = df['test_err'] < grok_threshold
    if not grok_mask.any():
        return np.nan
    first_grok = df[grok_mask]['step'].iloc[0]
    after = df[df['step'] >= first_grok]
    if len(after) < min_stable:
        return np.nan
    if not (after['test_err'].iloc[:min_stable] < grok_threshold).all():
        return np.nan
    if not (df['train_loss'] < train_loss_thresh).any():
        return np.nan
    return first_grok

# --- Adjust the base directory if needed ---
# If your runs/ folder is in the current working directory, use '.'.
# If it's somewhere else (e.g., /kaggle/input/...), change this variable.
BASE_DIR = "."   # change to your runs parent directory if necessary

# Find all log_seed*.csv files recursively
log_files = glob.glob(os.path.join(BASE_DIR, "runs/**/log_seed*.csv"), recursive=True)
print(f"Found {len(log_files)} log files.")
if len(log_files) == 0:
    print("No log files found. Check BASE_DIR and your folder structure.")
else:
    data = []
    for path in log_files:
        parts = path.split(os.sep)
        # Determine sweep type from path components
        if 'dataset_sweep' in path:
            sweep = 'dataset'
            dir_name = parts[-2]
            match = re.search(r'n_(\d+)_seed_(\d+)', dir_name)
            if match:
                param = int(match.group(1))
                seed = int(match.group(2))
        elif 'lambda_sweep' in path:
            sweep = 'lambda'
            dir_name = parts[-2]
            match = re.search(r'wd_(\d+\.?\d*)_seed_(\d+)', dir_name)
            if match:
                param = float(match.group(1))
                seed = int(match.group(2))
        elif 'arrhenius' in path:
            sweep = 'arrhenius'
            dir_name = parts[-2]
            match = re.search(r'lr_(\d+\.?\d*)_seed_(\d+)', dir_name)
            if match:
                param = float(match.group(1))
                seed = int(match.group(2))
        else:
            continue
        tau = get_tau_grok_corrected(path)
        data.append({'sweep': sweep, 'param': param, 'seed': seed, 'tau_grok': tau})
        print(f"{sweep} param={param} seed={seed} -> tau={tau}")

    df = pd.DataFrame(data)
    df.to_csv("final_output/all_sweeps_reanalysed.csv", index=False)

    # Generate figures per sweep
    batch_size = 512

    for sweep in df['sweep'].unique():
        subset = df[df['sweep'] == sweep].copy()
        if sweep == 'arrhenius':
            # Arrhenius plot: scatter + regression
            subset = subset[subset['tau_grok'].notna()]
            if subset.empty:
                print("No valid grokking for Arrhenius sweep")
                continue
            subset['inv_T'] = batch_size / subset['param']   # B/η
            subset['log_tau'] = np.log(subset['tau_grok'])
            slope, intercept, r_value, p_value, std_err = stats.linregress(subset['inv_T'], subset['log_tau'])
            print(f"\n=== Arrhenius sweep ===\nlog(tau) = {intercept:.4f} + {slope:.6f} * (B/lr)\nR² = {r_value**2:.4f}\np = {p_value:.2e}")
            plt.figure(figsize=(6,5))
            plt.scatter(subset['inv_T'], subset['log_tau'], alpha=0.6, label='Data')
            x_line = np.linspace(subset['inv_T'].min(), subset['inv_T'].max(), 100)
            plt.plot(x_line, intercept + slope * x_line, 'r-', label=f'fit: slope={slope:.6f}')
            plt.xlabel('B / lr (batch size / learning rate)')
            plt.ylabel('log(tau_grok)')
            plt.title('Arrhenius scaling for modular addition (transformer)')
            plt.legend()
            plt.grid(True)
            plt.savefig('final_output/arrhenius_corrected.png', dpi=150)
            plt.show()
        else:
            # Dataset or lambda sweep: errorbar plot (median ± IQR)
            summary = subset.groupby('param').agg(
                median_tau=('tau_grok', 'median'),
                q25=('tau_grok', lambda x: x.quantile(0.25)),
                q75=('tau_grok', lambda x: x.quantile(0.75))
            ).reset_index()
            valid = summary[summary['median_tau'].notna()].copy()
            if valid.empty:
                print(f"No valid grokking for {sweep} sweep")
                continue
            valid['yerr_low'] = valid['median_tau'] - valid['q25']
            valid['yerr_high'] = valid['q75'] - valid['median_tau']
            plt.figure(figsize=(8,5))
            plt.errorbar(valid['param'], valid['median_tau'], 
                         yerr=[valid['yerr_low'], valid['yerr_high']],
                         fmt='o-', capsize=5)
            xlabel = 'Dataset size n' if sweep == 'dataset' else 'Weight decay λ'
            title = f'{sweep.capitalize()} sweep (corrected detection)'
            plt.xlabel(xlabel)
            plt.ylabel('Grokking time τ (steps)')
            plt.title(title)
            plt.grid(True)
            plt.savefig(f'final_output/{sweep}_sweep_corrected.png', dpi=150)
            plt.show()

    # Separate by sweep type into individual CSV files
    for sweep in df['sweep'].unique():
        subset = df[df['sweep'] == sweep]
        subset.to_csv(f"final_output/{sweep}_corrected.csv", index=False)
        print(f"Saved {sweep}_corrected.csv with {len(subset)} rows")
