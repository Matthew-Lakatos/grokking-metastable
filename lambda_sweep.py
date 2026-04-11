# %% [markdown]
# # Lambda sweep for modular addition (transformer) – test regularisation effect
# # Runnable as is

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
n = 4000
batch = 512
lr = 0.002                # fixed learning rate (proven to grok)
lambdas = [0.1, 0.2, 0.3, 0.4]   # weight decay values
seeds = [0, 1, 2]
max_steps = 50000         # safe upper bound
log_interval = 100
grok_threshold = 0.1

master_path = "runs/lambda_sweep_master.csv"
results = []
completed = set()
if os.path.exists(master_path):
    existing = pd.read_csv(master_path)
    for _, row in existing.iterrows():
        completed.add((row['wd'], row['seed']))
    results = existing.to_dict('records')
    print(f"Resuming: {len(completed)} runs already completed.")

total = len(lambdas) * len(seeds)
run_idx = len(completed)

for wd, seed in product(lambdas, seeds):
    if (wd, seed) in completed:
        continue
    run_idx += 1
    outdir = f"runs/lambda_sweep/wd_{wd}_seed_{seed}"
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        "python", "run_experiment.py",
        "--task", task,
        "--model", model,
        "--n", str(n),
        "--batch", str(batch),
        "--wd", str(wd),
        "--lr", str(lr),
        "--seed", str(seed),
        "--outdir", outdir,
        "--max_steps", str(max_steps),
        "--log_interval", str(log_interval),
        "--grok_threshold", str(grok_threshold)
    ]
    print(f"[{run_idx}/{total}] wd={wd} seed={seed}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        tau = np.nan
    else:
        log_path = os.path.join(outdir, f"log_seed{seed}.csv")
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            grok_mask = df['test_err'] < grok_threshold
            if grok_mask.any():
                tau = df[grok_mask]['step'].iloc[0]
                print(f"  -> tau_grok = {tau}")
            else:
                tau = np.nan
                print(f"  -> No grokking")
    results.append({'wd': wd, 'seed': seed, 'tau_grok': tau})
    pd.DataFrame(results).to_csv(master_path, index=False)

# Summary
df = pd.read_csv(master_path)
summary = df.groupby('wd').agg(
    median_tau=('tau_grok', 'median'),
    grokked_ratio=('tau_grok', lambda x: (~np.isnan(x)).mean())
).reset_index()
print("\n=== Lambda sweep summary ===")
print(summary)
summary.to_csv('runs/lambda_summary.csv', index=False)

# Plot
plt.figure(figsize=(6,5))
plt.errorbar(summary['wd'], summary['median_tau'], fmt='o-')
plt.xlabel('Weight decay λ')
plt.ylabel('Median grokking time τ')
plt.title('Regularisation effect on grokking time')
plt.grid(True)
plt.savefig('runs/lambda_effect.png', dpi=150)
plt.show()
