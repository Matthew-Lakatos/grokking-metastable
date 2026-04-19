# Grokking as Metastable Complexity Dynamics

**Author:** Matthew Lakatos  
**Contact:** m.atthew.lakatos1@gmail.com

## Overview

This repository contains the complete code and analysis for the paper
*"Grokking as Metastable Complexity Dynamics"*. Grokking — the phenomenon
where a model suddenly generalises long after memorising training data — is
formalised as a metastable escape in a complexity-regularised free-energy
landscape.

The code implements:

- **Modular addition** (p = 128) as the primary task, using a small transformer.
- All order parameters defined in the paper: complexity C_norm and C_PB,
  alignment m(t), precision q(t), and test error ε_test(t).
- Effective temperature T_eff estimated via **FlucDis-SGD** (gradient-difference method).
- Geometric diagnostics: participation ratio (intrinsic dimensionality) and top
  Hessian eigenvalues via Lanczos iteration.
- Geometry checkpoints at **pre-transition**, **at-transition**, and
  **post-transition**.
- Full-domain evaluation for modular addition (16 384 pairs).
- **Arrhenius scaling** analysis (log τ vs B/lr) with linear regression.

The code is self-contained, uses only PyTorch and standard scientific Python
libraries, and is designed for full reproducibility.

---

## Repository Structure

```
grokking-metastable/
├── run_experiment.py           # Main training script
├── run_full_sweep.sh           # Runs all experiments then generates figures
├── requirements.txt
├── diagnostics/
│   ├── __init__.py
│   ├── geometry.py             # Participation ratio and Hessian eigenvalues
│   └── order_params.py         # Order parameters + shared get_tau_grok()
├── experiments/
│   ├── sweep_runner.py         # Arrhenius sweep (varies learning rate)
│   ├── lambda_sweep.py         # Weight-decay sweep
│   ├── dataset_sweep.py        # Dataset-size sweep
│   └── causal_test.py          # Causal LR-switch experiment
├── analysis/
│   ├── fit_arrhenius.py        # Standalone Arrhenius fit (CLI)
│   ├── fit_precision.py        # Precision-reallocation figure (CLI)
│   └── phase_diagram.py        # Phase-diagram heatmaps (CLI)
├── reproducibility/
│   └── reproduce.sh            # Quick smoke test
├── final_output/
│   └── analyser.py             # Authoritative figure/result generator
└── data/
    └── ...                     # Pre-generated paper figures and CSVs
```

`runs/` is created at runtime and holds all intermediate logs, checkpoints,
and diagnostic plots. `final_output/` is populated by `analyser.py` and
contains the paper-quality figures.

---

## Requirements

```
Python 3.8+
PyTorch ≥ 1.12
NumPy, Pandas, Matplotlib, SciPy, PyYAML, tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Quick Start (Smoke Test)

Verifies that the transformer groks on modular addition (single run,
~7 000 steps to grokking):

```bash
chmod +x reproducibility/reproduce.sh
./reproducibility/reproduce.sh
```

Expected output: `runs/smoke_transformer/` containing a CSV log and three
geometry checkpoints (`.npz`). The log should show `test_err = 0.000` around
step 7 000.

---

## Full Experiments

### Option A — one command

```bash
chmod +x run_full_sweep.sh
./run_full_sweep.sh
```

This runs all four sweeps sequentially and then calls `final_output/analyser.py`
to produce the paper figures.

### Option B — individual sweeps

All scripts must be run from the **repository root**.

| Command | Description |
|---|---|
| `python experiments/sweep_runner.py` | Arrhenius sweep (5 LRs × 3 seeds) |
| `python experiments/lambda_sweep.py` | Weight-decay sweep (4 λ × 3 seeds) |
| `python experiments/dataset_sweep.py` | Dataset-size sweep (7 sizes × 3 seeds) |
| `python experiments/causal_test.py` | Causal LR-switch experiment |
| `python final_output/analyser.py` | Generate all paper figures |

Each sweep script resumes automatically if interrupted.

---

## Arrhenius Sweep Configuration

```
Task:                modular addition (p = 128)
Model:               tiny transformer (2 layers, 2 heads, embedding dim 32)
Fixed hyperparams:   n = 4000, batch = 512, weight_decay = 0.3, log_interval = 25
Learning rates:      0.0005, 0.001, 0.002, 0.004, 0.008
Seeds:               0, 1, 2
Max steps per LR:    0.0005 → 150 000 | 0.001 → 100 000 | others → 50 000
```

Runtime on a single NVIDIA T4 GPU: approximately 6–8 hours. All scripts are
resumable across multiple sessions.

---

## Outputs and Metrics

Each training run writes to its `--outdir`:

| File | Description |
|---|---|
| `log_seed{seed}.csv` | Step-wise metrics: `step, time, train_loss, C_norm, C_PB, m, q_logit, q_ent, test_err, hess_top, PR, T_eff_proxy` |
| `checkpoint.pt` | Model and optimiser state for resuming |
| `geometry_pre.npz` | Geometry snapshot at step 0 |
| `geometry_at.npz` | Geometry snapshot at first grokking event |
| `geometry_post.npz` | Geometry snapshot at end of training |

`final_output/analyser.py` reads all logs and produces:

| File | Description |
|---|---|
| `final_output/all_sweeps_reanalysed.csv` | Combined τ_grok for all sweeps |
| `final_output/arrhenius_corrected.png` | Arrhenius plot (log τ vs B/lr) |
| `final_output/dataset_corrected.png` | Dataset-size sweep error-bar plot |
| `final_output/lambda_corrected.png` | Lambda sweep error-bar plot |
| `final_output/{sweep}_corrected.csv` | Per-sweep τ_grok values |

---

## Grokking Detection

All scripts share the single authoritative implementation `get_tau_grok()` in
`diagnostics/order_params.py`. A run is considered to have grokked at the
first step *t* where:

1. `test_err < 0.1` at step *t*.
2. `test_err` stays below `0.1` for **5 consecutive** log entries (stability guard).
3. `train_loss` drops below **0.5** at *any* point in the log (sanity check).

---

## Reproducibility

- All random seeds are fixed (`torch.manual_seed`, `np.random.seed`).
- Evaluation uses the full domain — no sampling noise.
- All sweep scripts save results incrementally and resume after interruption.

---

## Optional Analysis Scripts

```bash
# Standalone Arrhenius fit (reads sweep_runner output directly):
python analysis/fit_arrhenius.py --master runs/arrhenius_transformer_master.csv

# Precision-reallocation figure for a single log:
python analysis/fit_precision.py --log runs/arrhenius_transformer/lr_0.002_seed_0/log_seed0.csv

# Phase diagrams (requires a master_results.csv with task/lambda/n columns):
python analysis/phase_diagram.py --master runs/master_results.csv
```

---

## Citation

```bibtex
@article{lakatos2026grokking,
  title   = {Grokking as Metastable Complexity Dynamics},
  author  = {Lakatos, Matthew},
  journal = {},
  year    = {2026}
}
```

---

## License

MIT License — free for academic use.
