# Grokking as Metastable Complexity Dynamics

**Author:** Matthew Lakatos  
**Contact:** m.atthew.lakatos1@gmail.com

Special thanks to Deepseek AI for generating the README.md

## Overview

This repository contains the complete code and analysis for the paper  
*“Grokking as Metastable Complexity Dynamics”*. Grokking is formalised – the phenomenon where a model suddenly generalises long after memorising the training data – as a metastable escape in a complexity‑regularised free‑energy landscape. The code implements:

- Two algorithmic tasks: **modular addition** (p=128) and **sparse parity** (16‑bit inputs, parity over first 3 bits).
- All order parameters defined in the paper: complexity \(C_{\mathrm{norm}}\) and \(C_{\mathrm{PB}}\), alignment \(m(t)\), precision \(q(t)\), test error \(\epsilon_{\mathrm{test}}(t)\).
- Effective temperature \(T_{\mathrm{eff}}\) estimated from SGD noise (gradient variance, learning rate, batch size).
- Geometric diagnostics: top‑5 Hessian eigenvalues (Lanczos) and participation ratio (intrinsic dimensionality).
- Geometry checkpoints saved at **pre‑transition**, **at‑transition**, and **post‑transition**.
- Full‑domain evaluation for both tasks (16384 pairs for modular addition, 8 patterns for sparse parity).
- Arrhenius scaling analysis and phase diagrams.

The code is self‑contained, uses only PyTorch and standard scientific Python libraries, and is designed for full reproducibility.

## Repository Structure
```text
grokking-metastable/
├── README.md
├── requirements.txt
├── run_experiment.py # Main training script (all tasks, all metrics)
├── diagnostics/
│ ├── init.py
│ ├── geometry.py # Hessian (Lanczos) and participation ratio
│ └── order_params.py # C_norm, C_PB, alignment, precision, test error
├── configs/
│ ├── modular_addition.yaml # Example config (not used directly, kept for reference)
│ └── sparse_parity.yaml
├── reproducibility/
│ └── reproduce.sh # Smoke tests for both tasks
├── experiments/
│ └── sweep_runner.py # Full experiment sweeps (both tasks, 5 seeds)
├── analysis/
│ ├── fit_arrhenius.py # Arrhenius scaling from master_results.csv
│ └── phase_diagram.py # Phase diagrams (λ vs n)
└── runs/ # Created at runtime; stores logs, checkpoints, results
```

## Requirements

```text
- Python 3.8+
- PyTorch ≥ 1.12
- NumPy, Pandas, Matplotlib, SciPy, PyYAML, tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

Quick Start (Smoke Tests)
Run the smoke tests to verify that everything works. This will run short experiments (2000 steps) for both modular addition and sparse parity.

# Make the script executable

```bash
chmod +x reproducibility/reproduce.sh
```

# Run both smoke tests

```bash
./reproducibility/reproduce.sh
```

Expected output: two folders runs/smoke_modular/ and runs/smoke_sparse/ each containing a CSV log (log_seed0.csv) and geometry checkpoints (geometry_pre.npz, geometry_post.npz). No errors should appear.

Full Experiments (Reproducing Paper Results)
To reproduce the main results of the paper, run the full sweep over:

Tasks: modular_add, sparse_parity

Regularisation λ (weight decay): 1e-3, 1e-2, 1e-1

Dataset size n: 500, 1000

Batch size: 64

Seeds: 0,1,2,3,4 (5 seeds)

Max steps: 50,000 for n=500, 100,000 for n=1000

Execute:

```bash
python experiments/sweep_runner.py
```

On a single NVIDIA T4 GPU, the full sweep takes approximately 10‑12 hours. Results are saved incrementally to runs/master_results.csv. The sweep automatically resumes from where it left off if interrupted.

After the sweep completes, generate the paper’s figures:

```bash
python analysis/fit_arrhenius.py        # Arrhenius plot (log τ vs 1/T_eff)
python analysis/phase_diagram.py        # Heatmap of median grokking time
Outputs are saved in the runs/ directory.
```

Individual Run Example
To run a single experiment (e.g., modular addition with λ=1e-2, n=500, seed=0):

```bash
python run_experiment.py \
    --task modular_add \
    --model tiny_mlp \
    --n 500 \
    --batch 64 \
    --wd 1e-2 \
    --seed 0 \
    --outdir runs/my_test \
    --max_steps 50000 \
    --log_interval 200 \
    --grok_threshold 0.1
All command‑line arguments are documented inside run_experiment.py.
```

Outputs and Metrics
Each run produces:

CSV log (log_seed{seed}.csv): columns – step, time, train_loss, C_norm, C_PB, m, q_logit, q_ent, test_err, hess_top, PR, T_eff_proxy.

Geometry checkpoints (.npz files):

geometry_pre.npz – at step 0.

geometry_at.npz – the first time test_err < grok_threshold (if grokking occurs).

geometry_post.npz – at the end of training.
Each checkpoint contains hess_top5 (list of top‑5 Hessian eigenvalues) and participation_ratio.

Reproducibility Guarantee
All random seeds are fixed (torch.manual_seed, np.random.seed).

The same train/test splits are used (full domain evaluation, no randomness in test set).

The code is deterministic (no operations with nondeterministic behaviour).

The sweep runner saves intermediate results, allowing resumption after interruption.

Customisation
You can easily add new tasks by subclassing torch.utils.data.Dataset and extending make_dataloaders and the evaluation block in run_experiment.py. The order parameters and geometric diagnostics work for any classification task.

Citation
If you use this code in your research, please cite the paper:

```bibtex
@article{lakatos2026grokking,
  title={Grokking as Metastable Complexity Dynamics},
  author={Lakatos, Matthew},
  journal={ ... },
  year={2026}
}
```

License
MIT License – feel free to use and adapt for academic purposes.
