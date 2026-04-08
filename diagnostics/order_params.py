import math, torch, torch.nn.functional as F
import numpy as np

def compute_C_norm(model):
    s = 0.0
    for p in model.parameters():
        s += (p.detach()**2).sum().item()
    return 0.5 * s

def compute_C_PB(model, sigma_p=1.0, sigma_q=1e-5):
    theta_sq = 0.0; d = 0
    for p in model.parameters():
        theta_sq += (p.detach()**2).sum().item()
        d += p.numel()
    ratio = (sigma_q**2) / (sigma_p**2)
    kl = 0.5 * (theta_sq / (sigma_p**2) + d * (ratio - 1.0 - math.log(max(ratio,1e-30))))
    return kl

def compute_alignment(logits_eval, canonical_logits):
    if canonical_logits is None:
        return 0.0
    a = logits_eval.view(logits_eval.shape[0], -1)
    b = canonical_logits.view(canonical_logits.shape[0], -1)
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    num = (a * b).sum().item()
    den = ( (a**2).sum().item() * (b**2).sum().item() )**0.5 + 1e-12
    return num / den

def compute_precision(logits_eval):
    logits = logits_eval.detach()
    q_logit = float(logits.std(dim=-1).mean().item())
    probs = F.softmax(logits, dim=-1)
    ent = -(probs * (probs + 1e-12).log()).sum(dim=-1).mean().item()
    return q_logit, -ent

def evaluate_test_error(model, X_eval, Y_eval):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(X_eval.to(device))
        preds = logits.argmax(dim=-1).cpu()
        err = (preds != Y_eval).float().mean().item()
    return err
