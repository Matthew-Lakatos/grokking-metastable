#!/usr/bin/env python3
"""
experiments/causal_test.py
Causal test: effect of learning rate on grokking time.

Three conditions are compared across 3 seeds:
  - Constant low LR  (lr=0.0002, 100 000 steps)
  - Constant high LR (lr=0.002,  100 000 steps)
  - Low → high switch (lr=0.0002 for 40 000 steps, then lr=0.002 for 30 000 more)

Outputs:
  runs/causality/causality_plot.png  – Mean ± std test-error curves.
  runs/causality/causality_summary.csv – Per-seed grokking times.

Run from the repository root:
    python experiments/causal_test.py
"""

import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostics.order_params import get_tau_grok

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TASK         = "modular_add"
MODEL        = "tiny_transformer"
N            = 4000
BATCH        = 512
WEIGHT_DECAY = 0.3
LOG_INTERVAL = 25
GROK_THRESHOLD = 0.1

LOW_LR  = 0.0002
HIGH_LR = 0.002
SEEDS   = [0, 1, 2]

CONSTANT_STEPS = 100_000
SWITCH_PHASE1  = 40_000     # Steps at low LR before the switch.
SWITCH_TOTAL   = 70_000     # Total steps for the switch condition.

OUT_DIR = "runs/causality"


# ---------------------------------------------------------------------------
# Completion check
# ---------------------------------------------------------------------------

def run_is_complete(outdir, seed, required_steps):
    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    if not os.path.exists(log_path):
        return False
    df = pd.read_csv(log_path)
    return int(df['step'].max()) >= required_steps - 1


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def _run_cmd(cmd, label):
    """Run a subprocess command and print a minimal status line."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {label}: {result.stderr.strip()}")
    return result.returncode == 0


def run_constant(seed, lr, max_steps, label):
    """Run a constant-LR condition; skip if already complete."""
    outdir = os.path.join(OUT_DIR, f"constant_{label}_seed{seed}")
    os.makedirs(outdir, exist_ok=True)
    if run_is_complete(outdir, seed, max_steps):
        print(f"  Constant {label} seed={seed} — already complete, skipping.")
        return outdir
    cmd = [
        "python", "run_experiment.py",
        "--task", TASK, "--model", MODEL,
        "--n", str(N), "--batch", str(BATCH), "--wd", str(WEIGHT_DECAY),
        "--lr", str(lr), "--seed", str(seed),
        "--outdir", outdir,
        "--max_steps", str(max_steps),
        "--log_interval", str(LOG_INTERVAL),
        "--grok_threshold", str(GROK_THRESHOLD),
    ]
    print(f"  Constant {label} (lr={lr})  seed={seed}")
    _run_cmd(cmd, f"constant_{label}_seed{seed}")
    return outdir


def run_switch(seed):
    """
    Run the low→high switch condition.

    Phase 1: train at LOW_LR for SWITCH_PHASE1 steps.
    Phase 2: resume at HIGH_LR until SWITCH_TOTAL steps.
    """
    outdir = os.path.join(OUT_DIR, f"switch_low2high_seed{seed}")
    os.makedirs(outdir, exist_ok=True)
    if run_is_complete(outdir, seed, SWITCH_TOTAL):
        print(f"  Switch low→high seed={seed} — already complete, skipping.")
        return outdir

    base_args = [
        "--task", TASK, "--model", MODEL,
        "--n", str(N), "--batch", str(BATCH), "--wd", str(WEIGHT_DECAY),
        "--seed", str(seed), "--outdir", outdir,
        "--log_interval", str(LOG_INTERVAL),
        "--grok_threshold", str(GROK_THRESHOLD),
    ]
    # Phase 1.
    print(f"  Switch low→high seed={seed}  phase 1 (lr={LOW_LR})")
    _run_cmd(
        ["python", "run_experiment.py"] + base_args +
        ["--lr", str(LOW_LR), "--max_steps", str(SWITCH_PHASE1)],
        f"switch_phase1_seed{seed}",
    )
    # Phase 2 (resume).
    print(f"  Switch low→high seed={seed}  phase 2 (lr={HIGH_LR})")
    _run_cmd(
        ["python", "run_experiment.py"] + base_args +
        ["--lr", str(HIGH_LR), "--max_steps", str(SWITCH_TOTAL),
         "--resume", "--resume_from", os.path.join(outdir, "checkpoint.pt")],
        f"switch_phase2_seed{seed}",
    )
    return outdir


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_test_error(outdir, seed):
    """Return (steps, test_err) arrays from the log CSV, or (None, None)."""
    path = os.path.join(outdir, f"log_seed{seed}.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    return df['step'].values, df['test_err'].values


def aggregate_curves(seed_outdir_pairs, max_steps):
    """
    Interpolate each run's test-error curve onto a common step grid and
    return (steps, mean, std).
    """
    common = np.arange(0, max_steps + 1, LOG_INTERVAL)
    all_errors = []
    for seed, outdir in seed_outdir_pairs:
        steps, err = load_test_error(outdir, seed)
        if steps is None:
            continue
        interp = np.interp(common[common <= steps[-1]], steps, err)
        full = np.full(len(common), np.nan)
        full[:len(interp)] = interp
        all_errors.append(full)

    if not all_errors:
        nan_arr = np.full(len(common), np.nan)
        return common, nan_arr, nan_arr

    arr = np.array(all_errors)
    return common, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Run all conditions.
    constant_low_dirs  = []
    constant_high_dirs = []
    switch_dirs        = []

    for seed in SEEDS:
        constant_low_dirs.append((seed,  run_constant(seed, LOW_LR,  CONSTANT_STEPS, "low")))
        constant_high_dirs.append((seed, run_constant(seed, HIGH_LR, CONSTANT_STEPS, "high")))
        switch_dirs.append((seed, run_switch(seed)))

    # Aggregate curves.
    steps_low,    mean_low,    std_low    = aggregate_curves(constant_low_dirs,  CONSTANT_STEPS)
    steps_high,   mean_high,   std_high   = aggregate_curves(constant_high_dirs, CONSTANT_STEPS)
    steps_switch, mean_switch, std_switch = aggregate_curves(switch_dirs,        SWITCH_TOTAL)

    # Plot.
    plt.figure(figsize=(10, 6))
    plt.plot(steps_low,    mean_low,    'b-', label=f'Constant low LR ({LOW_LR})',  linewidth=2)
    plt.fill_between(steps_low,  mean_low  - std_low,  mean_low  + std_low,  color='b', alpha=0.2)
    plt.plot(steps_high,   mean_high,   'r-', label=f'Constant high LR ({HIGH_LR})', linewidth=2)
    plt.fill_between(steps_high, mean_high - std_high, mean_high + std_high, color='r', alpha=0.2)
    plt.plot(steps_switch, mean_switch, 'g-', label='Low → high switch',            linewidth=2)
    plt.fill_between(steps_switch, mean_switch - std_switch, mean_switch + std_switch,
                     color='g', alpha=0.2)
    plt.axvline(x=SWITCH_PHASE1, color='k', linestyle='--',
                label=f'Switch point ({SWITCH_PHASE1 // 1000}k steps)')
    plt.xlabel('Step')
    plt.ylabel('Test error')
    plt.title('Causal test: effect of learning rate on grokking')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "causality_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path!r}.")

    # Summary CSV.
    rows = []
    for label, dirs in [("constant_low",  constant_low_dirs),
                         ("constant_high", constant_high_dirs),
                         ("low2high",      switch_dirs)]:
        for seed, outdir in dirs:
            log_path = os.path.join(outdir, f"log_seed{seed}.csv")
            tau = get_tau_grok(log_path, grok_threshold=GROK_THRESHOLD)
            rows.append({'condition': label, 'seed': seed, 'tau_grok': tau})

    summary_path = os.path.join(OUT_DIR, "causality_summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path!r}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
