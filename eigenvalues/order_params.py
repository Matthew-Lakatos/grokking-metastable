"""
diagnostics/order_params.py
Order parameters and grokking-detection utilities used throughout the codebase.

Public API
----------
get_tau_grok          – Authoritative grokking-time detector (shared by all sweep scripts).
compute_C_norm        – L2-norm complexity.
compute_C_PB          – PAC-Bayes KL complexity.
compute_alignment     – Cosine alignment between model logits and canonical logits.
compute_precision     – Logit standard deviation and negative entropy.
evaluate_test_error   – Full-domain classification error.
"""

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Grokking detection
# ---------------------------------------------------------------------------

def get_tau_grok(log_path, grok_threshold=0.1, train_loss_thresh=0.5, min_stable=5):
    """
    Detect grokking time from a training log CSV produced by run_experiment.py.

    A run is considered to have grokked at the first step *t* where all three
    conditions hold:

    1. ``test_err < grok_threshold`` at step *t*.
    2. ``test_err`` stays below ``grok_threshold`` for ``min_stable``
       consecutive log entries starting at *t* (stability guard).
    3. ``train_loss`` drops below ``train_loss_thresh`` at *any* point in the
       full log (sanity check that the model fitted the training data).

    Parameters
    ----------
    log_path : str
        Path to a ``log_seed*.csv`` file.
    grok_threshold : float
        Test-error threshold (default 0.1).
    train_loss_thresh : float
        Training loss must reach below this value at some point (default 0.5).
    min_stable : int
        Number of consecutive log entries for which test_err must remain below
        ``grok_threshold`` (default 5).

    Returns
    -------
    float or np.nan
        Training step at which grokking is detected, or ``np.nan`` if not.
    """
    if not os.path.exists(log_path):
        return np.nan
    df = pd.read_csv(log_path)
    if 'test_err' not in df.columns or 'train_loss' not in df.columns:
        return np.nan

    grok_mask = df['test_err'] < grok_threshold
    if not grok_mask.any():
        return np.nan

    first_grok_idx = grok_mask.idxmax()
    first_grok_step = df.loc[first_grok_idx, 'step']

    # Stability: min_stable consecutive entries must stay below threshold.
    after = df.loc[first_grok_idx:]
    if len(after) < min_stable:
        return np.nan
    if not (after['test_err'].iloc[:min_stable] < grok_threshold).all():
        return np.nan

    # Sanity: training loss must have reached a low value at some point.
    if not (df['train_loss'].dropna() < train_loss_thresh).any():
        return np.nan

    return float(first_grok_step)


# ---------------------------------------------------------------------------
# Complexity measures
# ---------------------------------------------------------------------------

def compute_C_norm(model):
    """L2-norm complexity: C_norm = 0.5 * ||theta||^2."""
    s = 0.0
    for p in model.parameters():
        s += (p.detach() ** 2).sum().item()
    return 0.5 * s


def compute_C_PB(model, sigma_p=1.0, sigma_q=1e-5):
    """
    PAC-Bayes KL complexity.

    KL( N(theta, sigma_q^2 I) || N(0, sigma_p^2 I) )
    """
    theta_sq = 0.0
    d = 0
    for p in model.parameters():
        theta_sq += (p.detach() ** 2).sum().item()
        d += p.numel()
    ratio = (sigma_q ** 2) / (sigma_p ** 2)
    kl = 0.5 * (
        theta_sq / (sigma_p ** 2)
        + d * (ratio - 1.0 - math.log(max(ratio, 1e-30)))
    )
    return kl


# ---------------------------------------------------------------------------
# Alignment and precision
# ---------------------------------------------------------------------------

def compute_alignment(logits_eval, canonical_logits):
    """
    Cosine alignment between model logits and canonical (ground-truth) logits,
    both mean-centred across the evaluation domain.
    """
    if canonical_logits is None:
        return 0.0
    a = logits_eval.view(logits_eval.shape[0], -1)
    b = canonical_logits.view(canonical_logits.shape[0], -1)
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    num = (a * b).sum().item()
    den = ((a ** 2).sum().item() * (b ** 2).sum().item()) ** 0.5 + 1e-12
    return num / den


def compute_precision(logits_eval):
    """
    Two precision proxies computed from the full-domain logit matrix.

    Returns
    -------
    q_logit : float
        Mean per-example logit standard deviation.
    q_ent : float
        Mean negative entropy (higher = sharper / more confident predictions).
    """
    logits = logits_eval.detach()
    q_logit = float(logits.std(dim=-1).mean().item())
    probs = F.softmax(logits, dim=-1)
    ent = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean().item()
    return q_logit, -ent


# ---------------------------------------------------------------------------
# Test error
# ---------------------------------------------------------------------------

def evaluate_test_error(model, X_eval, Y_eval):
    """Full-domain classification error (fraction of incorrect predictions)."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(X_eval.to(device))
        preds = logits.argmax(dim=-1).cpu()
        err = (preds != Y_eval).float().mean().item()
    return err
