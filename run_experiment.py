#!/usr/bin/env python3
"""
run_experiment.py
Supports modular addition and parity tasks with full domain evaluation,
canonical alignment, Hessian top‑k, participation ratio, gradient‑variance
logging, and geometry checkpoints.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diagnostics.geometry import lanczos_top_k, participation_ratio_from_model
from diagnostics.order_params import (
    compute_C_norm,
    compute_C_PB,
    compute_alignment,
    compute_precision,
    evaluate_test_error,
)

# creating datasets...
class ModularAdditionDataset(Dataset):
    def __init__(self, n_bits=7, n_samples=None):
        self.mod = 2 ** n_bits
        self.inputs = []
        self.targets = []
        domain = [(a, b) for a in range(self.mod) for b in range(self.mod)]
        if n_samples is None:
            pairs = domain
        else:
            rng = np.random.default_rng(0)
            pairs = rng.choice(len(domain), size=n_samples, replace=False)
            pairs = [domain[i] for i in pairs]
        for a, b in pairs:
            self.inputs.append([a, b])
            self.targets.append((a + b) % self.mod)
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class ParityDataset(Dataset):
    def __init__(self, n_bits=16, n_samples=1000):
        rng = np.random.default_rng(0)
        self.X = rng.integers(0, 2, size=(n_samples, n_bits)).astype(np.float32)
        self.y = (self.X.sum(axis=1) % 2).astype(np.int64)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# models
class TinyMLP(nn.Module):
    def __init__(self, input_dim, hidden=256, num_classes=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, emb=32, nhead=2, nlayers=2, num_classes=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pool = nn.Linear(emb, num_classes if num_classes is not None else vocab_size)
    def forward(self, x):
        e = self.emb(x).permute(1,0,2)
        h = self.encoder(e)
        h = h.mean(dim=0)
        return self.pool(h)

# utils
def make_dataloaders(task, n, batch_size):
    if task == "modular_add":
        ds = ModularAdditionDataset(n_bits=7, n_samples=n)
        def collate(batch):
            xs = torch.tensor([b[0] for b in batch], dtype=torch.long)
            ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
            return xs, ys
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
        return loader, ds
    elif task == "parity":
        ds = ParityDataset(n_bits=16, n_samples=n)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        return loader, ds
    else:
        raise ValueError("Unknown task")

def generate_full_parity_domain(n_bits=16):
    """All 2^n_bits bitstrings and their XOR parity labels."""
    num_samples = 2 ** n_bits
    X = np.zeros((num_samples, n_bits), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)
    for i in range(num_samples):
        bits = [(i >> b) & 1 for b in range(n_bits)]
        X[i] = bits
        y[i] = sum(bits) % 2
    return torch.from_numpy(X), torch.from_numpy(y)

def canonical_parity_logits(X):
    """Canonical logits: one‑hot of true parity (2 classes)."""
    # X is full domain tensor; we need labels. Simpler: recompute y from bits.
    n_bits = X.shape[1]
    y = torch.zeros(X.shape[0], dtype=torch.long)
    for i in range(X.shape[0]):
        bits = X[i].byte().tolist()
        y[i] = sum(bits) % 2
    return F.one_hot(y, num_classes=2).float()

def compute_grad_variance_from_params(model):
    """Variance of all parameter gradients (flattened)."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    if not grads:
        return 0.0
    flat_grads = torch.cat(grads)
    return flat_grads.var().item()

def save_geometry_checkpoint(model, criterion, X_sample, Y_sample, device, outdir, suffix):
    """Save top‑5 Hessian eigenvalues and participation ratio."""
    os.makedirs(outdir, exist_ok=True)
    n_sample = min(256, len(X_sample))
    X_sub = X_sample[:n_sample].to(device)
    Y_sub = Y_sample[:n_sample].to(device)
    try:
        hess_top5 = lanczos_top_k(model, criterion, X_sub, Y_sub, k=5, n_iter=50)
    except Exception:
        hess_top5 = [float('nan')] * 5
    try:
        pr = participation_ratio_from_model(model, X_sub, layer_names=['fc2', 'fc1', 'encoder', 'pool'])
    except Exception:
        pr = float('nan')
    np.savez(os.path.join(outdir, f'geometry_{suffix}.npz'),
             hess_top5=hess_top5,
             participation_ratio=pr,
             step=suffix)
    print(f"Saved geometry checkpoint: {suffix}")

# training...
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"log_seed{args.seed}.csv")

    # Build data loaders
    train_loader, train_ds = make_dataloaders(args.task, args.n, args.batch)

    # Build evaluation sets (full domain for both tasks)
    if args.task == "modular_add":
        full_ds = ModularAdditionDataset(n_bits=7, n_samples=None)
        X_list = np.array([x for x, _ in full_ds], dtype=np.int64)
        Y_list = np.array([y for _, y in full_ds], dtype=np.int64)
        X_eval = torch.from_numpy(X_list)
        Y_eval = torch.from_numpy(Y_list)
        canonical_logits = F.one_hot(Y_eval, num_classes=2**7).float()
        input_dim = 2  # two integers a,b
        num_classes = 2**7
    elif args.task == "parity":
        X_eval, Y_eval = generate_full_parity_domain(n_bits=16)
        canonical_logits = canonical_parity_logits(X_eval)
        input_dim = 16
        num_classes = 2
    else:
        raise ValueError("Unknown task")

    # Build model
    if args.model == "tiny_mlp":
        model = TinyMLP(input_dim=input_dim, hidden=args.hidden, num_classes=num_classes)
    elif args.model == "tiny_transformer":
        if args.task == "modular_add":
            model = TinyTransformer(vocab_size=2**7, emb=args.emb, num_classes=2**7)
        else:
            raise NotImplementedError("Transformer only for modular_add in this version")
    else:
        raise ValueError("Unknown model")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    # CSV header
    header = ["step","time","train_loss","C_norm","C_PB","m","q_logit","q_ent","test_err","hess_top","PR","T_eff_proxy"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    step = 0
    start_time = time.time()
    has_grokked = False

    # Initial logging (step 0)
    model.eval()
    with torch.no_grad():
        C_norm = compute_C_norm(model)
        C_PB = compute_C_PB(model, sigma_p=args.sigma_p, sigma_q=args.sigma_q)
        # Compute logits over X_eval in batches
        logits_eval = []
        bs = 256
        for i in range(0, len(X_eval), bs):
            xbatch = X_eval[i:i+bs].to(device)
            if args.model == "tiny_mlp":
                # For parity, X_eval is already float; for modular addition it's long
                if args.task == "modular_add":
                    xbatch = xbatch.float()
                logits = model(xbatch)
            else:  # transformer expects long
                logits = model(xbatch.long())
            logits_eval.append(logits.cpu())
        logits_eval = torch.cat(logits_eval, dim=0)
        m = compute_alignment(logits_eval, canonical_logits) if canonical_logits is not None else 0.0
        q_logit, q_ent = compute_precision(logits_eval)
        test_err = evaluate_test_error(model, X_eval, Y_eval)
        # Hessian top
        try:
            hess_top5 = lanczos_top_k(model, criterion, X_eval[:256].to(device), Y_eval[:256].to(device), k=5)
            hess_top = hess_top5[0]
        except Exception:
            hess_top = float('nan')
        # Participation ratio
        try:
            pr = participation_ratio_from_model(model, X_eval[:512].to(device))
        except Exception:
            pr = float('nan')
        T_eff_proxy = 0.0
        elapsed = time.time() - start_time
        row = [step, f"{elapsed:.1f}", float('nan'), C_norm, C_PB, m, q_logit, q_ent, test_err, hess_top, pr, T_eff_proxy]
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"[step {step}] test_err={test_err:.3f} (initial)")

    save_geometry_checkpoint(model, criterion, X_eval, Y_eval, device, args.outdir, 'pre')

    # Training loop
    while step < args.max_steps:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            if args.task == "modular_add":
                xs, ys = batch
                xs = xs.to(device)
                ys = ys.to(device)
                if args.model == "tiny_mlp":
                    inp = xs.float()
                    logits = model(inp)
                else:
                    logits = model(xs)
            else:  # parity
                xs, ys = batch
                xs = xs.to(device)
                ys = ys.to(device)
                logits = model(xs)   # xs already float
            loss = criterion(logits, ys)
            loss.backward()
            grad_var = compute_grad_variance_from_params(model)
            optimizer.step()
            step += 1

            if step % args.log_interval == 0:
                model.eval()
                with torch.no_grad():
                    C_norm = compute_C_norm(model)
                    C_PB = compute_C_PB(model, sigma_p=args.sigma_p, sigma_q=args.sigma_q)
                    logits_eval = []
                    bs = 256
                    for i in range(0, len(X_eval), bs):
                        xbatch = X_eval[i:i+bs].to(device)
                        if args.model == "tiny_mlp":
                            if args.task == "modular_add":
                                xbatch = xbatch.float()
                            logits = model(xbatch)
                        else:
                            logits = model(xbatch.long())
                        logits_eval.append(logits.cpu())
                    logits_eval = torch.cat(logits_eval, dim=0)
                    m = compute_alignment(logits_eval, canonical_logits) if canonical_logits is not None else 0.0
                    q_logit, q_ent = compute_precision(logits_eval)
                    test_err = evaluate_test_error(model, X_eval, Y_eval)
                    try:
                        hess_top5 = lanczos_top_k(model, criterion, X_eval[:256].to(device), Y_eval[:256].to(device), k=5)
                        hess_top = hess_top5[0]
                    except Exception:
                        hess_top = float('nan')
                    try:
                        pr = participation_ratio_from_model(model, X_eval[:512].to(device))
                    except Exception:
                        pr = float('nan')
                    T_eff_proxy = args.lr * grad_var / args.batch
                elapsed = time.time() - start_time
                row = [step, f"{elapsed:.1f}", float(loss.item()), C_norm, C_PB, m, q_logit, q_ent, test_err, hess_top, pr, T_eff_proxy]
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f"[step {step}] loss={loss.item():.4f} test_err={test_err:.3f} T_eff={T_eff_proxy:.3e}")

                if not has_grokked and test_err < args.grok_threshold:
                    save_geometry_checkpoint(model, criterion, X_eval, Y_eval, device, args.outdir, 'at')
                    has_grokked = True

            if step >= args.max_steps:
                break

    save_geometry_checkpoint(model, criterion, X_eval, Y_eval, device, args.outdir, 'post')
    print("Training finished. Logs saved to", csv_path)

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="modular_add", choices=["modular_add","parity"])
    parser.add_argument("--model", type=str, default="tiny_mlp", choices=["tiny_mlp","tiny_transformer"])
    parser.add_argument("--n", type=int, default=500, help="Number of training samples")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay (lambda)")
    parser.add_argument("--sigma_p", type=float, default=1.0)
    parser.add_argument("--sigma_q", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--emb", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--grok_threshold", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
