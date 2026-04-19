"""
diagnostics/geometry.py
Geometric diagnostics: participation ratio and Hessian top eigenvalues.

Public API
----------
participation_ratio            – PR from an activation matrix.
participation_ratio_from_model – PR via a forward-hook on a named layer.
lanczos_top_k                  – Top-k Hessian eigenvalues via Lanczos iteration.
lanczos_top_eig                – Single top eigenvalue via power iteration (fallback).
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Participation ratio
# ---------------------------------------------------------------------------

def participation_ratio(activations):
    """
    Participation ratio of an activation matrix.

    PR = (sum lambda_i)^2 / (sum lambda_i^2)

    where lambda_i are the eigenvalues of the sample covariance of
    *activations*. A high PR indicates spread activation (low effective
    dimensionality), a low PR indicates concentrated activation.

    Parameters
    ----------
    activations : np.ndarray, shape (n_samples, ...)
        Raw activations; flattened to (n_samples, d) internally.

    Returns
    -------
    float or np.nan
    """
    A = activations.reshape(activations.shape[0], -1)
    cov = np.cov(A, rowvar=False)
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.maximum(eigs, 0.0)
    s1 = eigs.sum()
    s2 = (eigs ** 2).sum()
    if s2 <= 0 or s1 <= 0:
        return float('nan')
    return float((s1 ** 2) / s2)


def participation_ratio_from_model(model, x, layer_names=None):
    """
    Compute the participation ratio by hooking the first available named layer.

    Tries each name in *layer_names* in order; falls back to the first
    ``nn.Linear`` module found if none match.

    Parameters
    ----------
    model : nn.Module
    x : torch.Tensor
        Input batch.
    layer_names : list[str] or None
        Ordered list of attribute names to try (default: ['fc2', 'fc1',
        'encoder', 'pool']).

    Returns
    -------
    float or np.nan
    """
    if layer_names is None:
        layer_names = ['fc2', 'fc1', 'encoder', 'pool']

    activations = []

    def hook(module, inp, out):
        activations.append(out.detach())

    handle = None
    for name in layer_names:
        if hasattr(model, name):
            handle = getattr(model, name).register_forward_hook(hook)
            break

    if handle is None:
        # Fallback: first Linear layer.
        for module in model.modules():
            if isinstance(module, nn.Linear):
                handle = module.register_forward_hook(hook)
                break

    if handle is None:
        return float('nan')

    with torch.no_grad():
        _ = model(x)
    handle.remove()

    if not activations:
        return float('nan')

    act = activations[0].cpu().numpy()
    return participation_ratio(act)


# ---------------------------------------------------------------------------
# Hessian eigenvalues
# ---------------------------------------------------------------------------

def lanczos_top_eig(model, loss_fn, X_sample, Y_sample, k=1, iters=20):
    """
    Power-iteration HVP fallback for the top Hessian eigenvalue.

    Kept for compatibility; ``lanczos_top_k`` is preferred.
    """
    device = next(model.parameters()).device
    model.zero_grad()
    logits = model(X_sample.to(device))
    loss = loss_fn(logits, Y_sample.to(device)) + 1e-12  # small damping
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)

    v = torch.randn(n, device=device, dtype=torch.float64)
    v = v / (v.norm() + 1e-12)

    for _ in range(iters):
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads]).double()
        Hv = torch.autograd.grad((flat_grads * v).sum(), params, retain_graph=True)
        Hv_flat = torch.cat([h.contiguous().view(-1).double() for h in Hv]).detach()
        v = Hv_flat / (Hv_flat.norm() + 1e-12)

    # Rayleigh quotient.
    grads = torch.autograd.grad(loss, params, create_graph=False)
    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads]).double()
    Hv = torch.autograd.grad((flat_grads * v).sum(), params, retain_graph=False)
    Hv_flat = torch.cat([h.contiguous().view(-1).double() for h in Hv]).detach()
    eig = (v * Hv_flat).sum().item()
    return [0.0 if abs(eig) < 1e-12 else eig]


def lanczos_top_k(model, loss_fn, X_sample, Y_sample, k=5, n_iter=50):
    """
    Top-k Hessian eigenvalues via Lanczos iteration (double precision).

    Parameters
    ----------
    model : nn.Module
    loss_fn : callable
    X_sample, Y_sample : torch.Tensor
        Mini-batch used to evaluate the Hessian.
    k : int
        Number of eigenvalues to return.
    n_iter : int
        Lanczos iterations.

    Returns
    -------
    list[float]
        Top-k eigenvalues in descending order, padded with 0.0 if fewer
        than k converge.
    """
    device = next(model.parameters()).device
    model.zero_grad()
    logits = model(X_sample.to(device))
    loss = loss_fn(logits, Y_sample.to(device)) + 1e-12  # small damping
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    def hvp(v):
        """Hessian-vector product in double precision."""
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads]).double()
        grad_v = (flat_grads * v.double()).sum()
        hv = torch.autograd.grad(grad_v, params, retain_graph=True)
        return torch.cat([h.contiguous().view(-1).double() for h in hv]).detach()

    # Lanczos recurrence.
    q = torch.randn(n_params, device=device, dtype=torch.float64)
    q = q / torch.norm(q)
    alphas, betas = [], []
    v_prev = None

    for i in range(n_iter):
        w = hvp(q)
        if i > 0:
            w = w - betas[-1] * v_prev
        alpha = torch.dot(q, w)
        alphas.append(alpha.item())
        w = w - alpha * q
        if i < n_iter - 1:
            beta = torch.norm(w)
            if beta < 1e-12:
                break  # Hessian is numerically zero; terminate early.
            betas.append(beta.item())
            v_prev = q
            q = w / beta

    m = len(alphas)
    if m == 0:
        return [0.0] * k

    # Build symmetric tridiagonal matrix and diagonalise.
    T = torch.zeros((m, m), device=device, dtype=torch.float64)
    for i in range(m):
        T[i, i] = alphas[i]
        if i < m - 1 and i < len(betas):
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

    try:
        eigvals = torch.linalg.eigvalsh(T)
    except Exception:
        return lanczos_top_eig(model, loss_fn, X_sample, Y_sample, k=1, iters=50) * k

    top_vals, _ = torch.sort(eigvals.real, descending=True)
    result = top_vals[:k].cpu().tolist()
    if len(result) < k:
        result += [0.0] * (k - len(result))
    result = [0.0 if abs(x) < 1e-12 else x for x in result]
    return result
