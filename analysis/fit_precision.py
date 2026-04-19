#!/usr/bin/env python3
"""
analysis/fit_precision.py
Plot precision reallocation (q_logit and q_ent) around the grokking transition.

Reads a single log CSV and marks the grokking step with a vertical line.

Usage:
    python analysis/fit_precision.py --log runs/arrhenius_transformer/lr_0.002_seed_0/log_seed0.csv
    python analysis/fit_precision.py --log <path> --out <output.png> --grok_threshold 0.1

Run from the repository root.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from diagnostics.order_params import get_tau_grok


def main():
    parser = argparse.ArgumentParser(
        description="Plot precision reallocation around the grokking transition."
    )
    parser.add_argument(
        "--log", type=str,
        default="runs/arrhenius_transformer/lr_0.002_seed_0/log_seed0.csv",
        help="Path to a log_seed*.csv file.",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path for the figure (default: <log_dir>/precision_reallocation.png).",
    )
    parser.add_argument(
        "--grok_threshold", type=float, default=0.1,
        help="Test-error threshold used to identify the grokking step.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.log):
        raise FileNotFoundError(f"Log file not found: {args.log!r}")

    df = pd.read_csv(args.log)

    grok_step = get_tau_grok(args.log, grok_threshold=args.grok_threshold)
    if pd.isna(grok_step):
        print("Warning: grokking not detected in this log — "
              "the vertical line will be omitted.")

    plt.figure(figsize=(6, 5))
    plt.plot(df['step'], df['q_logit'], label='q_logit (logit std)',  color='blue')
    plt.plot(df['step'], df['q_ent'],   label='q_ent (neg entropy)',  color='orange')
    if not pd.isna(grok_step):
        plt.axvline(x=grok_step, color='r', linestyle='--', label='grokking step')
    plt.xlabel('Step')
    plt.ylabel('Precision')
    plt.title('Precision reallocation at grokking')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = args.out or os.path.join(os.path.dirname(args.log),
                                        "precision_reallocation.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Figure saved to {out_path!r}.")


if __name__ == "__main__":
    main()
