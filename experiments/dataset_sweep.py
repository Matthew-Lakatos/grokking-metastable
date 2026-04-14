# %% [markdown]
# # Dataset Size Sweep (Sparser Grid, Resume)
# - wide range of n values
# - 3 seeds
# - max_steps = 100000
# - log_interval = 25

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
max_steps = 100000
log_interval = 25
grok_threshold = 0.1
ns = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]   # grid
seeds = [0, 1, 2]

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

# Load existing results
results = []
completed = set()
results_file = "dataset_sweep_results.csv"
if os.path.exists(results_file):
    existing = pd.read_csv(results_file)
    for _, row in existing.iterrows():
        completed.add((row['n'], row['seed']))
    results = existing.to_dict('records')
    print(f"Resuming: {len(completed)} runs already completed.")

total = len(ns) * len(seeds)
run_idx = len(completed)

for n, seed in product(ns, seeds):
    if (n, seed) in completed:
        continue
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

    # Save incrementally
    pd.DataFrame(results).to_csv(results_file, index=False)

# Summary and plot
df_res = pd.read_csv(results_file)
summary = df_res.groupby('n').agg(
    median_tau=('tau_grok', 'median'),
    q25=('tau_grok', lambda x: x.quantile(0.25)),
    q75=('tau_grok', lambda x: x.quantile(0.75))
).reset_index()
summary['yerr_low'] = summary['median_tau'] - summary['q25']
summary['yerr_high'] = summary['q75'] - summary['median_tau']

plt.figure(figsize=(8,5))
plt.errorbar(summary['n'], summary['median_tau'], 
             yerr=[summary['yerr_low'], summary['yerr_high']],
             fmt='o-', capsize=5, capthick=1, elinewidth=1)
plt.xlabel('Training set size n')
plt.ylabel('Grokking time τ (steps)')
plt.title('Effect of dataset size on grokking time (median ± IQR, 3 seeds)')
plt.grid(True, alpha=0.3)
plt.savefig('dataset_size_effect.png', dpi=150)
plt.show()
print("Plot saved as dataset_size_effect.png")
