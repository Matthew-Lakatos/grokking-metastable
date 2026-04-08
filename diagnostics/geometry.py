import torch, numpy as np

def participation_ratio(activations):
    A = activations.reshape(activations.shape[0], -1)
    cov = np.cov(A, rowvar=False)
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.maximum(eigs, 0.0)
    s1 = eigs.sum(); s2 = (eigs**2).sum()
    if s2 <= 0 or s1 <= 0: return float('nan')
    return float((s1**2) / s2)

def lanczos_top_eig(model, loss_fn, X_sample, Y_sample, k=1, iters=20):
    # simple power-iteration HVP-based top eigenvalue (returns list)
    device = next(model.parameters()).device
    model.zero_grad()
    logits = model(X_sample.to(device))
    loss = loss_fn(logits, Y_sample.to(device))
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    v = torch.randn(n, device=device); v = v / (v.norm()+1e-12)
    for _ in range(iters):
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
        Hv = torch.autograd.grad((flat_grads * v).sum(), params, retain_graph=True)
        Hv_flat = torch.cat([h.contiguous().view(-1) for h in Hv]).detach()
        v = Hv_flat / (Hv_flat.norm()+1e-12)
    Hv = torch.autograd.grad(flat_grads, params, retain_graph=True)  # optional
    # compute Rayleigh quotient
    # fallback: return NaN if unstable
    return [float((v * Hv_flat).sum().item())]
