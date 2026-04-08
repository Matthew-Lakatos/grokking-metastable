import torch, numpy as np
import torch.nn as nn

def participation_ratio(activations):
    A = activations.reshape(activations.shape[0], -1)
    cov = np.cov(A, rowvar=False)
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.maximum(eigs, 0.0)
    s1 = eigs.sum(); s2 = (eigs**2).sum()
    if s2 <= 0 or s1 <= 0: return float('nan')
    return float((s1**2) / s2)

def participation_ratio_from_model(model, x, layer_names=None):
    """
    Extract activations from first available layer in layer_names.
    Default tries ['fc2', 'fc1', 'encoder', 'pool'].
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
        # Fallback: use first linear layer
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

def lanczos_top_k(model, loss_fn, X_sample, Y_sample, k=5, n_iter=50):
    """
    Compute top k eigenvalues of the Hessian using Lanczos iteration.
    Returns list of eigenvalues (largest first).
    """
    device = next(model.parameters()).device
    model.zero_grad()
    logits = model(X_sample.to(device))
    loss = loss_fn(logits, Y_sample.to(device))
    params = [p for p in model.parameters() if p.requires_grad]
    # Flatten parameters
    param_vector = torch.cat([p.view(-1) for p in params])
    n_params = param_vector.shape[0]
    
    def hvp(v):
        # Compute Hessian-vector product
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
        grad_v = (flat_grads * v).sum()
        hv = torch.autograd.grad(grad_v, params, retain_graph=True)
        flat_hv = torch.cat([h.contiguous().view(-1) for h in hv])
        return flat_hv.detach()
    
    # Lanczos
    q = torch.randn(n_params, device=device)
    q = q / torch.norm(q)
    Q = []
    alphas = []
    betas = []
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
            betas.append(beta.item())
            v_prev = q
            q = w / beta
        Q.append(q.clone())
    # Build tridiagonal matrix
    T = torch.zeros((n_iter, n_iter), device=device)
    for i in range(n_iter):
        T[i, i] = alphas[i]
        if i < n_iter - 1:
            T[i, i+1] = betas[i]
            T[i+1, i] = betas[i]
    eigvals, _ = torch.linalg.eig(T)
    eigvals = eigvals.real
    # Sort descending
    top_vals, _ = torch.sort(eigvals, descending=True)
    return top_vals[:k].cpu().tolist()
