# %% [markdown]
# # Dataset Size Sweep (Prediction 5)
# 
# Vary n = [2000, 4000, 8000] with fixed η=0.002, λ=0.3, batch=512.

# %% [code]
import os, subprocess, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

if not os.path.exists("grokking-metastable"):
    !git clone https://github.com/Matthew-Lakatos/grokking-metastable.git
os.chdir("grokking-metastable")

task = "modular_add"
model = "tiny_transformer"
batch = 512
wd = 0.3
lr = 0.002
max_steps = 50000
log_interval = 500
grok_threshold = 0.1
ns = [2000, 4000, 8000]
seeds = [0, 1]

def get_tau_grok(csv_path, grok_threshold=0.1, train_loss_thresh=0.1, min_residence=25):
    if not os.path.exists(csv_path):
        return np.nan
    df = pd.read_csv(csv_path)
    grok_mask = df['test_err'] < grok_threshold
    if not grok_mask.any():
        return np.nan
    first_grok = df[grok_mask]['step'].iloc[0]
    df_before = df[df['step'] <= first_grok]
    low_loss = df_before['train_loss'] < train_loss_thresh
    if not low_loss.any():
        return np.nan
    t0 = df_before[low_loss]['step'].iloc[0]
    if first_grok - t0 < min_residence:
        return np.nan
    return first_grok

results = []
for n, seed in product(ns, seeds):
    outdir = f"runs/dataset_sweep/n_{n}_seed_{seed}"
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        "python", "run_experiment.py",
        "--task", task, "--model", model,
        "--n", str(n), "--batch", str(batch), "--wd", str(wd),
        "--lr", str(lr), "--seed", str(seed),
        "--outdir", outdir, "--max_steps", str(max_steps),
        "--log_interval", str(log_interval), "--grok_threshold", str(grok_threshold)
    ]
    print(f"n={n}, seed={seed}")
    subprocess.run(cmd)
    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    tau = get_tau_grok(log_path)
    results.append({'n': n, 'seed': seed, 'tau_grok': tau})

df = pd.DataFrame(results)
summary = df.groupby('n')['tau_grok'].agg(['median', 'std']).reset_index()
plt.figure(figsize=(6,5))
plt.errorbar(summary['n'], summary['median'], yerr=summary['std'], fmt='o-', capsize=5)
plt.xlabel('Training set size n')
plt.ylabel('Grokking time τ (steps)')
plt.title('Effect of dataset size on grokking time')
plt.grid(True)
plt.savefig('dataset_size_effect.png', dpi=150)
plt.show()
