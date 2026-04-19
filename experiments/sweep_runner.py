#!/usr/bin/env python3
"""
experiments/sweep_runner.py
Arrhenius sweep: vary learning rate, hold all other hyperparameters fixed.

Runs 5 learning rates × 3 seeds = 15 configurations sequentially.
Results are saved incrementally to ``runs/arrhenius_transformer_master.csv``
and the script resumes automatically if interrupted.

After all runs finish a quick diagnostic Arrhenius fit (log τ vs T_eff_proxy)
is printed and saved to ``runs/arrhenius_transformer_diagnostic.png``.
The paper-quality figure is produced by ``final_output/analyser.py``.

Run from the repository root:
    python experiments/sweep_runner.py
"""

import os
import subprocess
import tarfile
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from diagnostics.order_params import get_tau_grok

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
TASK         = "modular_add"
MODEL        = "tiny_transformer"
N            = 4000
BATCH        = 512
WEIGHT_DECAY = 0.3
GROK_THRESHOLD = 0.1
LOG_INTERVAL = 25
SEEDS        = [0, 1, 2]

# Per-learning-rate step budgets (conservative upper bounds).
LR_MAXSTEPS = {
    0.0005: 150_000,
    0.001:  100_000,
    0.002:   50_000,
    0.004:   50_000,
    0.008:   50_000,
}
LEARNING_RATES = list(LR_MAXSTEPS.keys())

MASTER_CSV = "runs/arrhenius_transformer_master.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_run_complete(outdir, seed):
    """Return True if the run wrote both a log CSV and a post-geometry file."""
    return (
        os.path.exists(os.path.join(outdir, "geometry_post.npz")) and
        os.path.exists(os.path.join(outdir, f"log_seed{seed}.csv"))
    )


def get_T_eff_at_grok(log_path, tau_grok_step):
    """
    Return the T_eff_proxy value at the grokking step.

    Falls back to the median of the last 10 log entries before the
    transition if the exact step is not present.
    """
    if np.isnan(tau_grok_step) or not os.path.exists(log_path):
        return np.nan
    df = pd.read_csv(log_path)
    if 'T_eff_proxy' not in df.columns:
        return np.nan
    row = df[df['step'] == tau_grok_step]
    if not row.empty:
        return float(row['T_eff_proxy'].iloc[0])
    before = df[df['step'] <= tau_grok_step]
    if before.empty:
        return np.nan
    return float(before['T_eff_proxy'].iloc[-min(10, len(before)):].median())


def run_single(lr, seed, outdir, max_steps):
    """Launch run_experiment.py as a subprocess and return (tau_grok, T_eff)."""
    cmd = [
        "python", "run_experiment.py",
        "--task",           TASK,
        "--model",          MODEL,
        "--n",              str(N),
        "--batch",          str(BATCH),
        "--wd",             str(WEIGHT_DECAY),
        "--lr",             str(lr),
        "--seed",           str(seed),
        "--outdir",         outdir,
        "--max_steps",      str(max_steps),
        "--log_interval",   str(LOG_INTERVAL),
        "--grok_threshold", str(GROK_THRESHOLD),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr.strip()}")
        return np.nan, np.nan

    log_path = os.path.join(outdir, f"log_seed{seed}.csv")
    tau = get_tau_grok(log_path, grok_threshold=GROK_THRESHOLD)
    teff = get_T_eff_at_grok(log_path, tau)
    return tau, teff


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    os.makedirs("runs", exist_ok=True)

    # Resume: load already-completed (lr, seed) pairs.
    results = []
    completed = set()
    if os.path.exists(MASTER_CSV):
        existing = pd.read_csv(MASTER_CSV)
        for _, row in existing.iterrows():
            completed.add((row['lr'], row['seed']))
        results = existing.to_dict('records')
        print(f"Resuming — {len(completed)} run(s) already completed.")
    else:
        print("Starting new Arrhenius sweep.")

    total = len(LEARNING_RATES) * len(SEEDS)
    run_idx = len(completed)
    sweep_start = time.time()

    for lr in LEARNING_RATES:
        max_steps = LR_MAXSTEPS[lr]
        for seed in SEEDS:
            if (lr, seed) in completed:
                continue

            run_idx += 1
            outdir = f"runs/arrhenius_transformer/lr_{lr}_seed_{seed}"
            os.makedirs(outdir, exist_ok=True)

            print(f"\n[{run_idx}/{total}] lr={lr}  seed={seed}  "
                  f"max_steps={max_steps:,}")
            t0 = time.time()

            if is_run_complete(outdir, seed):
                log_path = os.path.join(outdir, f"log_seed{seed}.csv")
                tau  = get_tau_grok(log_path, grok_threshold=GROK_THRESHOLD)
                teff = get_T_eff_at_grok(log_path, tau)
                print(f"  Already complete — tau_grok={tau}, T_eff={teff:.3e}")
            else:
                tau, teff = run_single(lr, seed, outdir, max_steps)
                elapsed = time.time() - t0
                teff_str = f"{teff:.3e}" if not np.isnan(teff) else "nan"
                print(f"  tau_grok={tau}  T_eff={teff_str}  "
                      f"({elapsed / 60:.1f} min)")

                # Verification.
                log_path = os.path.join(outdir, f"log_seed{seed}.csv")
                if os.path.exists(log_path):
                    df = pd.read_csv(log_path)
                    min_err = df['test_err'].min()
                    has_teff = (df['T_eff_proxy'] > 0).any()
                    print(f"  [check] min_test_err={min_err:.4f}  "
                          f"T_eff_nonzero={has_teff}")
                    if min_err < GROK_THRESHOLD and np.isnan(tau):
                        print("  [WARNING] test_err dropped but grokking not "
                              "detected — check stability/train_loss criteria.")

            results.append({'lr': lr, 'seed': seed,
                            'tau_grok': tau, 'T_eff_proxy': teff})
            pd.DataFrame(results).to_csv(MASTER_CSV, index=False)

            total_elapsed = time.time() - sweep_start
            eta = (total - run_idx) * (total_elapsed / run_idx) if run_idx > 0 else 0
            print(f"  Elapsed: {total_elapsed / 3600:.2f} h  "
                  f"ETA: {eta / 3600:.2f} h")

    print(f"\nAll runs complete. Results saved to {MASTER_CSV!r}.")


# ---------------------------------------------------------------------------
# Diagnostic Arrhenius fit (uses T_eff_proxy from the log)
# ---------------------------------------------------------------------------

def diagnostic_arrhenius_fit():
    """
    Quick Arrhenius fit using the FlucDis T_eff_proxy column.

    This is a diagnostic check only.  The authoritative paper figure
    (using B/lr as the temperature proxy) is produced by
    final_output/analyser.py.
    """
    df = pd.read_csv(MASTER_CSV)
    valid = df[df['tau_grok'].notna() & (df['tau_grok'] > 0) &
               df['T_eff_proxy'].notna() & (df['T_eff_proxy'] > 0)].copy()

    if len(valid) < 5:
        print(f"Not enough valid runs for Arrhenius fit ({len(valid)} available).")
        return

    valid['log_tau'] = np.log(valid['tau_grok'])
    valid['inv_T']   = 1.0 / valid['T_eff_proxy']
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        valid['inv_T'], valid['log_tau']
    )

    print("\n" + "=" * 50)
    print("DIAGNOSTIC ARRHENIUS FIT (T_eff_proxy from FlucDis)")
    print("=" * 50)
    print(f"log(τ) = {intercept:.4f} + {slope:.4f} × (1/T_eff)")
    print(f"R²={r_value ** 2:.4f}  p={p_value:.2e}  se={std_err:.4f}")

    plt.figure(figsize=(6, 5))
    plt.scatter(valid['inv_T'], valid['log_tau'], alpha=0.6, label='Data')
    x_line = np.linspace(valid['inv_T'].min(), valid['inv_T'].max(), 100)
    plt.plot(x_line, intercept + slope * x_line, 'r-',
             label=f'fit: slope={slope:.3f}')
    plt.xlabel('1 / T_eff (FlucDis)')
    plt.ylabel('log(τ_grok)')
    plt.title('Diagnostic Arrhenius — modular addition (transformer)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = "runs/arrhenius_transformer_diagnostic.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Diagnostic plot saved to {out_path!r}.")

    # Summary by lr.
    summary = df.groupby('lr').agg(
        median_tau=('tau_grok', 'median'),
        grokked_ratio=('tau_grok', lambda x: (~np.isnan(x)).mean()),
    ).reset_index()
    summary_path = "runs/arrhenius_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to {summary_path!r}.")
    print(summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

def archive_runs():
    """Compress the runs/ directory into runs_archive.tar.gz."""
    archive_path = "runs_archive.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add("runs")
    print(f"Results archived to {archive_path!r}.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_sweep()
    diagnostic_arrhenius_fit()
    archive_runs()
