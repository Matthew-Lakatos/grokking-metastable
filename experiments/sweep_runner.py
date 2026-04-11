# %% [markdown]
# # Full Arrhenius Sweep – Modular Addition with Transformer
# 
# Vary learning rate, fix other hyperparameters.  
# Runs 5 learning rates × 3 seeds = 15 runs.  
# Uses per‑LR max steps based on observed grokking times.  
# Resumes automatically if interrupted.

# %% [code]
import os, subprocess, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import product

# ------------------------------------------------------------
# 1. Setup repository (clone if not present)
# ------------------------------------------------------------
if not os.path.exists("grokking-metastable"):
    !git clone https://github.com/Matthew-Lakatos/grokking-metastable.git
os.chdir("grokking-metastable")

# ------------------------------------------------------------
# 2. Experiment configuration
# ------------------------------------------------------------
task = "modular_add"
model = "tiny_transformer"

# Fixed hyperparameters
n = 4000
batch = 512
wd = 0.3                # weight decay (fixed)
grok_threshold = 0.1
log_interval = 100      # uniform for all runs (must satisfy definition)
seeds = [0, 1, 2]

# Learning rates and their step budgets (conservative upper bounds)
lr_maxsteps = {
    0.0005: 150000,
    0.001:  100000,
    0.002:   50000,
    0.004:   50000,
    0.008:   50000,
}
learning_rates = list(lr_maxsteps.keys())

master_path = "runs/arrhenius_transformer_master.csv"

# ------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------
def is_run_complete(outdir, seed, max_steps):
    """Check if a run completed successfully (final checkpoint exists)."""
    post_path = os.path.join(outdir, f"geometry_post.npz")
    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    if not os.path.exists(post_path) or not os.path.exists(log_path):
        return False
    return True

def get_tau_grok_and_teff(csv_path, grok_threshold=0.1, train_loss_thresh=0.1, min_residence=100):
    """
    Compute grokking time and effective temperature at transition.
    Now uses train_loss_thresh = 0.1 (instead of 1e-3) to accommodate transformer.
    """
    if not os.path.exists(csv_path):
        return np.nan, np.nan
    df = pd.read_csv(csv_path)
    if 'T_eff_proxy' not in df.columns or 'train_loss' not in df.columns:
        return np.nan, np.nan
    grok_mask = df['test_err'] < grok_threshold
    if not grok_mask.any():
        return np.nan, np.nan
    first_grok = df[grok_mask]['step'].iloc[0]
    df_before = df[df['step'] <= first_grok]
    low_loss_mask = df_before['train_loss'] < train_loss_thresh
    if not low_loss_mask.any():
        return np.nan, np.nan
    t0 = df_before[low_loss_mask]['step'].iloc[0]
    if first_grok - t0 < min_residence:
        return np.nan, np.nan
    teff_row = df[df['step'] == first_grok]
    if teff_row.empty:
        teff = df_before['T_eff_proxy'].median()
    else:
        teff = teff_row['T_eff_proxy'].iloc[0]
    return first_grok, teff

# ------------------------------------------------------------
# 4. Run sweep
# ------------------------------------------------------------
results = []
completed = set()

if os.path.exists(master_path):
    existing = pd.read_csv(master_path)
    for _, row in existing.iterrows():
        completed.add((row['lr'], row['seed']))
    results = existing.to_dict('records')
    print(f"Resuming: {len(completed)} runs already completed.")
else:
    print("Starting new sweep.")

total = len(learning_rates) * len(seeds)
run_idx = len(completed)
start_time = time.time()

for lr in learning_rates:
    max_steps = lr_maxsteps[lr]
    for seed in seeds:
        if (lr, seed) in completed:
            continue
        run_idx += 1
        outdir = f"runs/arrhenius_transformer/lr_{lr}_seed_{seed}"
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
        print(f"[{run_idx}/{total}] lr={lr} seed={seed} (max_steps={max_steps})")
        start_run = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_run = time.time() - start_run

        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            tau, teff = np.nan, np.nan
        else:
            log_path = os.path.join(outdir, f"log_seed{seed}.csv")
            tau, teff = get_tau_grok_and_teff(log_path, grok_threshold)  # uses new threshold
            print(f"  -> tau_grok = {tau}, T_eff = {teff:.3e} (time {elapsed_run/60:.1f} min)")

        results.append({
            'lr': lr,
            'seed': seed,
            'tau_grok': tau,
            'T_eff_proxy': teff
        })
        pd.DataFrame(results).to_csv(master_path, index=False)

        elapsed_total = time.time() - start_time
        remaining = (total - run_idx) * (elapsed_total / run_idx) if run_idx > 0 else 0
        print(f"  Total elapsed: {elapsed_total/3600:.2f}h, remaining: {remaining/3600:.2f}h")

print("\nAll runs completed. Results saved to", master_path)

# ------------------------------------------------------------
# 5. Arrhenius fit and plot
# ------------------------------------------------------------
df = pd.read_csv(master_path)
valid = df[(df['tau_grok'].notna()) & (df['tau_grok'] > 0) & (df['T_eff_proxy'] > 0)]

if len(valid) >= 5:
    valid['log_tau'] = np.log(valid['tau_grok'])
    valid['inv_T'] = 1.0 / valid['T_eff_proxy']
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid['inv_T'], valid['log_tau'])
    print("\n" + "="*50)
    print("ARRHENIUS FIT RESULTS")
    print("="*50)
    print(f"log(tau) = {intercept:.4f} + {slope:.4f} * (1/T_eff)")
    print(f"R² = {r_value**2:.4f}")
    print(f"p-value = {p_value:.2e}")
    print(f"Standard error = {std_err:.4f}")

    # Plot
    plt.figure(figsize=(6,5))
    plt.scatter(valid['inv_T'], valid['log_tau'], alpha=0.6, label='Data')
    x_line = np.linspace(valid['inv_T'].min(), valid['inv_T'].max(), 100)
    plt.plot(x_line, intercept + slope * x_line, 'r-', label=f'fit: slope={slope:.3f}')
    plt.xlabel('1 / T_eff (from FlucDis)')
    plt.ylabel('log(tau_grok)')
    plt.title('Arrhenius scaling – Modular addition (transformer)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('runs/arrhenius_transformer.png', dpi=150)
    plt.show()
else:
    print(f"Not enough valid runs (got {len(valid)}). Please check logs.")

# ------------------------------------------------------------
# 6. Save summary
# ------------------------------------------------------------
summary = df.groupby('lr').agg(
    median_tau=('tau_grok', 'median'),
    grokked_ratio=('tau_grok', lambda x: (~np.isnan(x)).mean())
).reset_index()
summary.to_csv('runs/arrhenius_summary.csv', index=False)
print("\nSummary statistics saved to runs/arrhenius_summary.csv")
print(summary)

# Archive results
!tar -czf runs_archive.tar.gz runs/
print("All results archived to runs_archive.tar.gz")
