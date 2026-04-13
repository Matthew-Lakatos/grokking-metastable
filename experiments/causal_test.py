# %% [markdown]
# # Causality Check (Corrected) – Single Graph, log_interval=25, 3 seeds
# - low LR = 0.0002, high LR = 0.002
# - Constant low, constant high, low→high switch
# - log_interval = 25
# - Seeds: 0,1,2

# %% [code]
import os, subprocess
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
wd = 0.3
log_interval = 25
grok_threshold = 0.1
low_lr = 0.0002
high_lr = 0.002
seeds = [0, 1, 2]

constant_steps = 100000
switch_phase1 = 40000
switch_total = 70000

# Helper to check if run already completed
def run_completed(outdir, seed, required_steps):
    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    if not os.path.exists(log_path):
        return False
    df = pd.read_csv(log_path)
    return df['step'].max() >= required_steps - 1

def run_constant(seed, lr, max_steps, label):
    outdir = f"runs/causality/constant_{label}_seed{seed}"
    os.makedirs(outdir, exist_ok=True)
    if run_completed(outdir, seed, max_steps):
        print(f"Constant {label} seed={seed} already completed. Skipping.")
        return outdir
    cmd = [
        "python", "run_experiment.py",
        "--task", task, "--model", model,
        "--n", str(n), "--batch", str(batch), "--wd", str(wd),
        "--lr", str(lr), "--seed", str(seed),
        "--outdir", outdir, "--max_steps", str(max_steps),
        "--log_interval", str(log_interval), "--grok_threshold", str(grok_threshold)
    ]
    print(f"Constant {label} (lr={lr}) seed={seed}")
    subprocess.run(cmd)
    return outdir

def run_switch(seed, first_lr, second_lr, phase1, total, label):
    outdir = f"runs/causality/switch_{label}_seed{seed}"
    os.makedirs(outdir, exist_ok=True)
    if run_completed(outdir, seed, total):
        print(f"Switch {label} seed={seed} already completed. Skipping.")
        return outdir
    # Phase 1
    cmd1 = [
        "python", "run_experiment.py",
        "--task", task, "--model", model,
        "--n", str(n), "--batch", str(batch), "--wd", str(wd),
        "--lr", str(first_lr), "--seed", str(seed),
        "--outdir", outdir, "--max_steps", str(phase1),
        "--log_interval", str(log_interval), "--grok_threshold", str(grok_threshold)
    ]
    print(f"Switch low→high seed={seed} phase1 (lr={first_lr})")
    subprocess.run(cmd1)
    # Phase 2 resume
    cmd2 = [
        "python", "run_experiment.py",
        "--task", task, "--model", model,
        "--n", str(n), "--batch", str(batch), "--wd", str(wd),
        "--lr", str(second_lr), "--seed", str(seed),
        "--outdir", outdir, "--max_steps", str(total),
        "--log_interval", str(log_interval), "--grok_threshold", str(grok_threshold),
        "--resume", "--resume_from", f"{outdir}/checkpoint.pt"
    ]
    print(f"Switch low→high seed={seed} phase2 (lr={second_lr})")
    subprocess.run(cmd2)
    return outdir

# Run experiments
constant_low_dirs = []
constant_high_dirs = []
switch_dirs = []

for seed in seeds:
    d_low = run_constant(seed, low_lr, constant_steps, "low")
    constant_low_dirs.append((seed, d_low))
    d_high = run_constant(seed, high_lr, constant_steps, "high")
    constant_high_dirs.append((seed, d_high))
    d_switch = run_switch(seed, low_lr, high_lr, switch_phase1, switch_total, "low2high")
    switch_dirs.append((seed, d_switch))

# Function to load test error and align to common steps
def load_test_error(outdir, seed):
    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    if not os.path.exists(log_path):
        return None, None
    df = pd.read_csv(log_path)
    return df['step'].values, df['test_err'].values

common_steps = np.arange(0, constant_steps + 1, log_interval)

def get_mean_std(dirs, max_steps):
    all_errors = []
    for seed, outdir in dirs:
        steps, err = load_test_error(outdir, seed)
        if steps is None:
            continue
        # Interpolate to common steps
        err_interp = np.interp(common_steps[common_steps <= steps[-1]], steps, err)
        full_err = np.full(len(common_steps), np.nan)
        full_err[:len(err_interp)] = err_interp
        all_errors.append(full_err)
    if not all_errors:
        return common_steps, np.full(len(common_steps), np.nan), np.full(len(common_steps), np.nan)
    all_errors = np.array(all_errors)
    mean_err = np.nanmean(all_errors, axis=0)
    std_err = np.nanstd(all_errors, axis=0)
    return common_steps, mean_err, std_err

steps_low, mean_low, std_low = get_mean_std(constant_low_dirs, constant_steps)
steps_high, mean_high, std_high = get_mean_std(constant_high_dirs, constant_steps)
steps_switch, mean_switch, std_switch = get_mean_std(switch_dirs, switch_total)

# Plot
plt.figure(figsize=(10,6))
plt.plot(steps_low, mean_low, 'b-', label='Constant low LR (0.0002)', linewidth=2)
plt.fill_between(steps_low, mean_low - std_low, mean_low + std_low, color='b', alpha=0.2)
plt.plot(steps_high, mean_high, 'r-', label='Constant high LR (0.002)', linewidth=2)
plt.fill_between(steps_high, mean_high - std_high, mean_high + std_high, color='r', alpha=0.2)
plt.plot(steps_switch, mean_switch, 'g-', label='Low→high switch', linewidth=2)
plt.fill_between(steps_switch, mean_switch - std_switch, mean_switch + std_switch, color='g', alpha=0.2)
plt.axvline(x=switch_phase1, color='k', linestyle='--', label='Switch point (40k steps)')
plt.xlabel('Step')
plt.ylabel('Test error')
plt.title('Causal test: effect of learning rate on grokking')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('causality_single.png', dpi=150)
plt.show()
print("Plot saved as causality_single.png")

# Generate summary CSV
summary_data = []
for seed, outdir in constant_low_dirs:
    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        grok = df[df['test_err'] < grok_threshold]
        tau = grok['step'].iloc[0] if not grok.empty else None
        summary_data.append({'condition': 'constant_low', 'seed': seed, 'tau_grok': tau})
for seed, outdir in constant_high_dirs:
    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        grok = df[df['test_err'] < grok_threshold]
        tau = grok['step'].iloc[0] if not grok.empty else None
        summary_data.append({'condition': 'constant_high', 'seed': seed, 'tau_grok': tau})
for seed, outdir in switch_dirs:
    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        grok = df[df['test_err'] < grok_threshold]
        tau = grok['step'].iloc[0] if not grok.empty else None
        summary_data.append({'condition': 'low2high', 'seed': seed, 'tau_grok': tau})
pd.DataFrame(summary_data).to_csv('causality_summary.csv', index=False)
print("Summary saved as causality_summary.csv")
