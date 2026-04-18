# Grokking as Metastable Complexity Dynamics

**Author:** Matthew Lakatos  
**Contact:** m.atthew.lakatos1@gmail.com  
**Special Thanks:** Deepseek AI for writing the README.md and code documentation

## Overview

This repository contains the complete code and analysis for the paper  
*“Grokking as Metastable Complexity Dynamics”*. Grokking – the phenomenon where a model suddenly generalises long after memorising the training data – is formalised as a metastable escape in a complexity‑regularised free‑energy landscape.

The code implements:

- **Modular addition** (p=128) as the primary task, using a small transformer.
- All order parameters defined in the paper: complexity \(C_{\mathrm{norm}}\) and \(C_{\mathrm{PB}}\), alignment \(m(t)\), precision \(q(t)\), test error \(\epsilon_{\mathrm{test}}(t)\).
- Effective temperature \(T_{\mathrm{eff}}\) estimated using **FlucDis‑SGD** (robust gradient‑difference method).
- Geometric diagnostics: participation ratio (intrinsic dimensionality).
- Geometry checkpoints saved at **pre‑transition**, **at‑transition**, and **post‑transition**.
- Full‑domain evaluation for modular addition (16 384 pairs).
- **Arrhenius scaling** analysis (log τ vs 1/T_eff) with linear regression.

The code is self‑contained, uses only PyTorch and standard scientific Python libraries, and is designed for full reproducibility.

## Repository Structure

```text
grokking-metastable/
├── README.md
├── requirements.txt
├── run_experiment.py          # Main training script (transformer, metrics, checkpoints)
├── diagnostics/
│   ├── __init__.py
│   ├── geometry.py            # Participation ratio (and Hessian, optional)
│   └── order_params.py        # Order parameters
├── reproducibility/
│   └── reproduce.sh           # Smoke test for modular addition (transformer)
├── experiments/
│   └── sweep_runner.py        # Full Arrhenius sweep (varies learning rate)
├── analysis/                  # Optional post‑processing scripts (not required)
│   ├── fit_arrhenius.py
│   └── phase_diagram.py
├── runs/                      # Created at runtime – IGNORE PLOTS AND TAU GROK - intermediate step for output
└── final_output/
    └── ...                    # Created at runtime - hosts the end results, figures and all data, ensures correctness
```

Requirements

```text
Python 3.8+

PyTorch ≥ 1.12

NumPy, Pandas, Matplotlib, SciPy, PyYAML, tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

Quick Start (Smoke Test)
Run a short smoke test to verify that the transformer groks on modular addition:

```bash
chmod +x reproducibility/reproduce.sh
./reproducibility/reproduce.sh
```

The smoke test runs a single configuration (n=4000, batch=512, wd=0.3, lr=0.002, max_steps=20000) with logging every 25 steps. Expected output: a folder runs/smoke_transformer/ containing a CSV log and geometry checkpoints. The test should finish without errors and show test_err = 0.000 after ~7000 steps.

Full Experiments (Reproducing the Paper Results)
The main experiment is an Arrhenius sweep that varies the learning rate (hence the effective temperature 
T
e
f
f
T 
eff
​
 ) while keeping all other hyperparameters fixed. It uses:

```text
Task: modular addition (p=128)

Model: tiny transformer (2 layers, 2 heads, embedding 32)

Fixed hyperparameters: n=4000, batch=512, weight_decay=0.3, log_interval=25

Learning rates: 0.0005, 0.001, 0.002, 0.004, 0.008

Seeds: 0,1,2 (3 seeds per learning rate)

Max steps (per learning rate, conservative upper bounds):

0.0005 → 150 000 steps

0.001 → 100 000 steps

0.002, 0.004, 0.008 → 50 000 steps
```

To run the full sweep:

```bash
python experiments/sweep_runner.py
```

The script:

Runs 15 configurations sequentially (5 LRs × 3 seeds).

Saves results incrementally to runs/arrhenius_transformer_master.csv.

Automatically resumes if interrupted (e.g., after a Kaggle session limit).

After all runs finish, it performs a linear regression of 
log
⁡
τ
grok
logτ 
grok
​
  vs 
1
/
T
e
f
f
1/T 
eff
​
  and saves the Arrhenius plot as runs/arrhenius_transformer.png.

Runtime: On a single NVIDIA T4 GPU, the full sweep takes approximately 6‑8 hours (due to fine logging interval of 25 steps). The script is resumable, so you can run it over multiple sessions if needed.

Outputs and Metrics
Each run produces:

```text
CSV log (log_seed{seed}.csv): columns – step, time, train_loss, C_norm, C_PB, m, q_logit, q_ent, test_err, PR, T_eff_proxy.

Geometry checkpoints (.npz):

geometry_pre.npz – at step 0.

geometry_at.npz – at the first step where test_err < 0.1.

geometry_post.npz – at the end of training.

The sweep script also produces arrhenius_transformer.png and a summary CSV with median grokking times.
```

Reproducibility Guarantee
All random seeds are fixed (torch.manual_seed, np.random.seed).

Evaluation uses the full domain (no sampling noise).

The code is deterministic.

The sweep runner saves intermediate results, allowing resumption after interruption.

Customisation
You can modify hyperparameters directly in sweep_runner.py. To add a new task, extend run_experiment.py (datasets, evaluation) – the order parameters and geometric diagnostics work for any classification task.

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
MIT License – free for academic use.
