# %% [markdown]
# # Learning Rate Switch Experiment (Causal Test)
# 
# Phase 1: low LR (0.0005) for 20k steps.  
# Phase 2: high LR (0.002) for another 10k steps, resuming from checkpoint.

# %% [code]
import os, subprocess
import pandas as pd
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
log_interval = 500
grok_threshold = 0.1
seed = 0
outdir = "runs/lr_switch"

# Phase 1: low learning rate
low_lr = 0.0005
phase1_steps = 20000
subprocess.run([
    "python", "run_experiment.py",
    "--task", task, "--model", model,
    "--n", str(n), "--batch", str(batch), "--wd", str(wd),
    "--lr", str(low_lr), "--seed", str(seed),
    "--outdir", outdir, "--max_steps", str(phase1_steps),
    "--log_interval", str(log_interval), "--grok_threshold", str(grok_threshold)
])

# Phase 2: resume with high learning rate
high_lr = 0.002
phase2_steps = 30000   # total steps (additional 10k)
subprocess.run([
    "python", "run_experiment.py",
    "--task", task, "--model", model,
    "--n", str(n), "--batch", str(batch), "--wd", str(wd),
    "--lr", str(high_lr), "--seed", str(seed),
    "--outdir", outdir, "--max_steps", str(phase2_steps),
    "--log_interval", str(log_interval), "--grok_threshold", str(grok_threshold),
    "--resume", "--resume_from", f"{outdir}/checkpoint.pt"
])

# Plot results
log_path = os.path.join(outdir, f"log_seed{seed}.csv")
if os.path.exists(log_path):
    df = pd.read_csv(log_path)
    plt.figure(figsize=(10,5))
    plt.plot(df['step'], df['test_err'], label='Test error')
    plt.axvline(x=phase1_steps, color='r', linestyle='--', label='LR switch (0.0005 → 0.002)')
    plt.xlabel('Step')
    plt.ylabel('Test error')
    plt.title('Learning rate switch experiment')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_switch_plot.png', dpi=150)
    plt.show()
else:
    print("Log file not found.")
