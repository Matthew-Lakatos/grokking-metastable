# %% [markdown]
# # Learning Rate Switch Experiment (Causal Test)
# 
# Tests causality: increasing temperature (LR) after memorisation should trigger grokking.
# Runs for seeds 0,1,2,3 and generates a combined plot.

# %% [code]
import os, subprocess, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists("grokking-metastable"):
    !git clone https://github.com/Matthew-Lakatos/grokking-metastable.git
os.chdir("grokking-metastable")

# Fixed hyperparameters
task = "modular_add"
model = "tiny_transformer"
n = 4000
batch = 512
wd = 0.3
log_interval = 25          # fine resolution
grok_threshold = 0.1
low_lr = 0.0005
high_lr = 0.002
phase1_steps = 20000
phase2_steps = 30000       # total steps after phase1 (additional 10k)
seeds = [0, 1, 2, 3]

# Store results for plotting
all_dfs = []
switch_step = phase1_steps

for seed in seeds:
    print(f"\n=== Running seed {seed} ===")
    outdir = f"runs/lr_switch/seed_{seed}"
    os.makedirs(outdir, exist_ok=True)

    # Phase 1: low learning rate
    cmd1 = [
        "python", "run_experiment.py",
        "--task", task, "--model", model,
        "--n", str(n), "--batch", str(batch), "--wd", str(wd),
        "--lr", str(low_lr), "--seed", str(seed),
        "--outdir", outdir, "--max_steps", str(phase1_steps),
        "--log_interval", str(log_interval), "--grok_threshold", str(grok_threshold)
    ]
    print("Phase 1 (low LR)...")
    subprocess.run(cmd1)

    # Phase 2: resume with high learning rate
    cmd2 = [
        "python", "run_experiment.py",
        "--task", task, "--model", model,
        "--n", str(n), "--batch", str(batch), "--wd", str(wd),
        "--lr", str(high_lr), "--seed", str(seed),
        "--outdir", outdir, "--max_steps", str(phase2_steps),
        "--log_interval", str(log_interval), "--grok_threshold", str(grok_threshold),
        "--resume", "--resume_from", f"{outdir}/checkpoint.pt"
    ]
    print("Phase 2 (high LR, resuming)...")
    subprocess.run(cmd2)

    # Load combined log
    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df['seed'] = seed
        all_dfs.append(df)
        # Check grokking after switch
        after_switch = df[df['step'] >= phase1_steps]
        grok_steps = after_switch[after_switch['test_err'] < grok_threshold]['step']
        if not grok_steps.empty:
            first_grok = grok_steps.iloc[0]
            print(f"  -> Grokking detected at step {first_grok} ({(first_grok - phase1_steps)} steps after switch)")
        else:
            print("  -> No grokking detected after switch")
    else:
        print(f"  -> Log file missing for seed {seed}")

# Plot all seeds together
plt.figure(figsize=(10,6))
for df in all_dfs:
    plt.plot(df['step'], df['test_err'], label=f"seed {df['seed'].iloc[0]}")
plt.axvline(x=switch_step, color='k', linestyle='--', linewidth=1, label='LR switch')
plt.xlabel('Step')
plt.ylabel('Test error')
plt.title('Learning rate switch experiment (causal test)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('lr_switch_combined.png', dpi=150)
plt.show()

print("\nPlot saved as lr_switch_combined.png")
