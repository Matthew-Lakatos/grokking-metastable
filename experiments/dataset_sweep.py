# %% [markdown]
# # Dataset Size Sweep (Prediction 5)
# 
# Tests the effect of training set size on grokking time.
# n = 2000, 4000, 6000, 8000, seeds = 0,1,2.

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
log_interval = 25
grok_threshold = 0.1
ns = [2000, 4000, 6000, 8000]
seeds = [0, 1, 2]

def get_tau_grok(csv_path, grok_threshold=0.1, train_loss_thresh=0.1, min_residence=25):
    """Compute grokking time using residence condition."""
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
total = len(ns) * len(seeds)
run_idx = 0

for n, seed in product(ns, seeds):
    run_idx += 1
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
    print(f"[{run_idx}/{total}] n={n}, seed={seed}")
    start = time.time()
    subprocess.run(cmd)
    elapsed = time.time() - start
    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    tau = get_tau_grok(log_path)
    results.append({'n': n, 'seed': seed, 'tau_grok': tau})
    print(f"  -> tau_grok = {tau} (time {elapsed/60:.1f} min)")

# Save raw data
df_results = pd.DataFrame(results)
df_results.to_csv("dataset_sweep_results.csv", index=False)
print("\nRaw results saved to dataset_sweep_results.csv")

# Summary and plot
summary = df_results.groupby('n').agg(
    median_tau=('tau_grok', 'median'),
    std_tau=('tau_grok', 'std')
).reset_index()

plt.figure(figsize=(6,5))
plt.errorbar(summary['n'], summary['median_tau'], yerr=summary['std_tau'],
             fmt='o-', capsize=5, capthick=1, elinewidth=1)
plt.xlabel('Training set size n')
plt.ylabel('Grokking time τ (steps)')
plt.title('Effect of dataset size on grokking time')
plt.grid(True, alpha=0.3)
plt.savefig('dataset_size_effect.png', dpi=150)
plt.show()

print("\nPlot saved as dataset_size_effect.png")
