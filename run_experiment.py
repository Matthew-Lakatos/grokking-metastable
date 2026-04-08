#!/usr/bin/env python3
"""
run_experiment.py
Minimal experiment runner for grokking diagnostics.
Modified to include gradient variance logging and geometry checkpointing.
"""

import argparse
import csv
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# CHANGED: import from geometry instead of order_params for Hessian & PR
from diagnostics.geometry import lanczos_top_eig, participation_ratio_from_model
# Keep original order_params imports
from diagnostics.order_params import (
    compute_C_norm,
    compute_C_PB,
    compute_alignment,
    compute_precision,
    evaluate_test_error,
)

# ... (keep all Dataset and Model class definitions exactly as they are) ...
# (ModularAdditionDataset, ParityDataset, TinyMLP, TinyTransformer, make_dataloaders unchanged)

# NEW: Helper for gradient variance using already computed gradients
def compute_grad_variance_from_params(model):
    """Return variance of all parameter gradients (flattened)."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    if not grads:
        return 0.0
    flat_grads = torch.cat(grads)
    return flat_grads.var().item()

# NEW: Geometry checkpoint saver
def save_geometry_checkpoint(model, criterion, X_sample, Y_sample, device, outdir, suffix):
    """Save Hessian top eigenvalue and participation ratio to .npz file."""
    os.makedirs(outdir, exist_ok=True)
    # Use subset to avoid OOM (same as in training loop)
    n_sample = min(256, len(X_sample))
    X_sub = X_sample[:n_sample].to(device)
    Y_sub = Y_sample[:n_sample].to(device)
    
    # Compute top Hessian eigenvalue (using existing lanczos_top_eig)
    try:
        hess_top = lanczos_top_eig(model, criterion, X_sub, Y_sub, k=1, iters=20)
        hess_top = float(hess_top[0]) if hess_top else float('nan')
    except Exception:
        hess_top = float('nan')
    
    # Compute participation ratio from activations (using fc2 layer for MLP)
    try:
        # For modular_add with TinyMLP, use 'fc2' layer; for transformer, adjust if needed
        pr = participation_ratio_from_model(model, X_sub, layer_name='fc2')
    except Exception:
        pr = float('nan')
    
    np.savez(os.path.join(outdir, f'geometry_{suffix}.npz'),
             hess_top=hess_top,
             participation_ratio=pr,
             step=suffix)   # suffix can be 'pre', 'at', 'post'
    print(f"Saved geometry checkpoint: {suffix}")

# Modify train() function
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"log_seed{args.seed}.csv")
    
    # Build data
    train_loader, train_ds = make_dataloaders(args.task, args.n, args.batch)
    
    # Build model (unchanged)
    if args.model == "tiny_mlp":
        if args.task == "parity":
            input_dim = train_ds.X.shape[1]
            model = TinyMLP(input_dim=input_dim, hidden=args.hidden, num_classes=2)
        else:
            model = TinyMLP(input_dim=2, hidden=args.hidden, num_classes=2**7)
    elif args.model == "tiny_transformer":
        if args.task == "modular_add":
            model = TinyTransformer(vocab_size=2**7, emb=args.emb, num_classes=2**7)
        else:
            raise NotImplementedError("Transformer only for modular_add")
    else:
        raise ValueError("Unknown model")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare evaluation grid (unchanged)
    if args.task == "modular_add":
        full_ds = ModularAdditionDataset(n_bits=7, n_samples=None)
        X_eval = torch.tensor([x for x, _ in full_ds], dtype=torch.long)
        Y_eval = torch.tensor([y for _, y in full_ds], dtype=torch.long)
        canonical_logits = F.one_hot(Y_eval, num_classes=2**7).float()
    else:
        X_eval = torch.tensor(train_ds.X, dtype=torch.float32)
        Y_eval = torch.tensor(train_ds.y, dtype=torch.long)
        canonical_logits = None
    
    # CHANGED: CSV header now includes T_eff_proxy
    header = ["step","time","train_loss","C_norm","C_PB","m","q_logit","q_ent","test_err","hess_top","PR","T_eff_proxy"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    
    step = 0
    start_time = time.time()
    has_grokked = False
    
    # NEW: Log initial state (step 0) and save pre-checkpoint
    model.eval()
    with torch.no_grad():
        # Compute initial diagnostics (same as in loop)
        C_norm = compute_C_norm(model).item()
        C_PB = compute_C_PB(model, sigma_p=args.sigma_p, sigma_q=args.sigma_q).item()
        # compute logits over X_eval
        logits_eval = []
        bs = 256
        for i in range(0, len(X_eval), bs):
            xbatch = X_eval[i:i+bs].to(device)
            if args.model == "tiny_mlp" and args.task == "modular_add":
                le = model(xbatch.float())
            else:
                le = model(xbatch)
            logits_eval.append(le.cpu())
        logits_eval = torch.cat(logits_eval, dim=0)
        m = compute_alignment(logits_eval, canonical_logits).item() if canonical_logits is not None else 0.0
        q_logit, q_ent = compute_precision(logits_eval)
        test_err = evaluate_test_error(model, X_eval, Y_eval)
        # Hessian & PR
        try:
            hess_top = lanczos_top_eig(model, criterion, X_eval[:256].to(device), Y_eval[:256].to(device), k=1)
            hess_top = float(hess_top[0])
        except:
            hess_top = float("nan")
        try:
            pr = participation_ratio_from_model(model, X_eval[:512].to(device), layer_name='fc2')
        except:
            pr = float("nan")
        # No gradient variance at step 0
        T_eff_proxy = 0.0
        elapsed = time.time() - start_time
        row = [step, f"{elapsed:.1f}", float('nan'), C_norm, C_PB, m, q_logit, q_ent, test_err, hess_top, pr, T_eff_proxy]
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"[step {step}] test_err={test_err:.3f} (initial)")
    
    # Save pre-checkpoint
    save_geometry_checkpoint(model, criterion, X_eval, Y_eval, device, args.outdir, 'pre')
    
    # Training loop
    while step < args.max_steps:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            # Forward pass (unchanged)
            if args.task == "modular_add":
                xs, ys = batch
                xs = xs.to(device)
                ys = ys.to(device)
                if args.model == "tiny_mlp":
                    inp = xs.float()
                    logits = model(inp)
                else:
                    logits = model(xs)
            else:
                xs, ys = batch
                xs = xs.to(device)
                ys = ys.to(device)
                logits = model(xs)
            loss = criterion(logits, ys)
            loss.backward()
            
            # NEW: Compute gradient variance BEFORE optimizer step
            grad_var = compute_grad_variance_from_params(model)
            
            optimizer.step()
            step += 1
            
            if step % args.log_interval == 0:
                model.eval()
                with torch.no_grad():
                    C_norm = compute_C_norm(model).item()
                    C_PB = compute_C_PB(model, sigma_p=args.sigma_p, sigma_q=args.sigma_q).item()
                    # compute logits over X_eval
                    logits_eval = []
                    bs = 256
                    for i in range(0, len(X_eval), bs):
                        xbatch = X_eval[i:i+bs].to(device)
                        if args.model == "tiny_mlp" and args.task == "modular_add":
                            le = model(xbatch.float())
                        else:
                            le = model(xbatch)
                        logits_eval.append(le.cpu())
                    logits_eval = torch.cat(logits_eval, dim=0)
                    m = compute_alignment(logits_eval, canonical_logits).item() if canonical_logits is not None else 0.0
                    q_logit, q_ent = compute_precision(logits_eval)
                    test_err = evaluate_test_error(model, X_eval, Y_eval)
                    try:
                        hess_top = lanczos_top_eig(model, criterion, X_eval[:256].to(device), Y_eval[:256].to(device), k=1)
                        hess_top = float(hess_top[0])
                    except:
                        hess_top = float("nan")
                    try:
                        pr = participation_ratio_from_model(model, X_eval[:512].to(device), layer_name='fc2')
                    except:
                        pr = float("nan")
                    # NEW: Compute T_eff_proxy
                    T_eff_proxy = args.lr * grad_var / args.batch_size
                
                elapsed = time.time() - start_time
                row = [step, f"{elapsed:.1f}", float(loss.item()), C_norm, C_PB, m, q_logit, q_ent, test_err, hess_top, pr, T_eff_proxy]
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f"[step {step}] loss={loss.item():.4f} test_err={test_err:.3f} T_eff={T_eff_proxy:.3e}")
                
                # NEW: Check for grokking (test error < 0.1) and save "at" checkpoint once
                if not has_grokked and test_err < 0.1:
                    save_geometry_checkpoint(model, criterion, X_eval, Y_eval, device, args.outdir, 'at')
                    has_grokked = True
            
            if step >= args.max_steps:
                break
        # end for batch
    # end while
    
    # NEW: Save post-checkpoint after training finishes
    save_geometry_checkpoint(model, criterion, X_eval, Y_eval, device, args.outdir, 'post')
    print("Training finished. Logs saved to", csv_path)

# CLI (unchanged)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="modular_add", choices=["modular_add","parity"])
    parser.add_argument("--model", type=str, default="tiny_mlp", choices=["tiny_mlp","tiny_transformer"])
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--sigma_p", type=float, default=1.0)
    parser.add_argument("--sigma_q", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--emb", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="runs")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
