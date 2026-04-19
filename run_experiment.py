#!/usr/bin/env python3
"""
run_experiment.py
Main training script for the grokking-metastable experiments.

Trains a TinyTransformer (modular addition) or TinyMLP (sparse parity or
modular addition) and logs order parameters, geometric diagnostics, and an
effective-temperature proxy at every ``--log_interval`` steps.

Supports resuming from a saved checkpoint (``--resume``).

Outputs (all written to ``--outdir``):
  log_seed<seed>.csv       – Step-wise metrics.
  checkpoint.pt            – Latest model and optimiser state.
  geometry_pre.npz         – Geometry at step 0.
  geometry_at.npz          – Geometry at first grokking event.
  geometry_post.npz        – Geometry at end of training.

All scripts must be run from the repository root.
"""

import argparse
import copy
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


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ModularAdditionDataset(Dataset):
    """
    Modular addition: (a + b) mod 2^n_bits, with a, b in [0, 2^n_bits).

    If n_samples is None, the full domain (2^n_bits)^2 pairs are used.
    Otherwise a random subset of size n_samples is drawn (seed=0).
    """

    def __init__(self, n_bits=7, n_samples=None):
        self.mod = 2 ** n_bits
        domain = [(a, b) for a in range(self.mod) for b in range(self.mod)]
        if n_samples is None:
            pairs = domain
        else:
            rng = np.random.default_rng(0)
            idxs = rng.choice(len(domain), size=n_samples, replace=False)
            pairs = [domain[i] for i in idxs]
        self.inputs = [[a, b] for a, b in pairs]
        self.targets = [(a + b) % self.mod for a, b in pairs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class SparseParityDataset(Dataset):
    """
    Sparse parity: label = XOR of the first active_bits bits of a random
    binary vector of length n_bits.
    """

    def __init__(self, n_bits=16, active_bits=3, n_samples=500):
        rng = np.random.default_rng(0)
        self.X = rng.integers(0, 2, size=(n_samples, n_bits)).astype(np.float32)
        self.y = (self.X[:, :active_bits].sum(axis=1) % 2).astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pool = nn.Linear(emb, num_classes if num_classes is not None else vocab_size)

    def forward(self, x):
        h = self.encoder(self.emb(x))
        return self.pool(h.mean(dim=1))


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def make_dataloaders(task, n, batch_size, active_bits=3):
    """Return (DataLoader, Dataset) for the requested task."""
    if task == "modular_add":
        ds = ModularAdditionDataset(n_bits=7, n_samples=n)

        def collate(batch):
            xs = torch.tensor([b[0] for b in batch], dtype=torch.long)
            ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
            return xs, ys

        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    elif task == "sparse_parity":
        ds = SparseParityDataset(n_bits=16, active_bits=active_bits, n_samples=n)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError(f"Unknown task: {task!r}")
    return loader, ds


def generate_full_sparse_parity_domain(active_bits=3, total_bits=16):
    """Return (X, y) covering all 2^active_bits patterns for sparse parity."""
    num_samples = 2 ** active_bits
    X = np.zeros((num_samples, total_bits), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)
    for i in range(num_samples):
        bits = [(i >> b) & 1 for b in range(active_bits)]
        X[i, :active_bits] = bits
        y[i] = sum(bits) % 2
    return torch.from_numpy(X), torch.from_numpy(y)


def canonical_sparse_parity_logits(X, active_bits=3):
    """One-hot canonical logits for the sparse parity task."""
    _, y = generate_full_sparse_parity_domain(active_bits, X.shape[1])
    return F.one_hot(y, num_classes=2).float()


# ---------------------------------------------------------------------------
# Effective temperature (FlucDis-SGD)
# ---------------------------------------------------------------------------

def compute_teff_flucdis(model, criterion, batch1, batch2, device, lr, batch_size,
                         task, num_classes_mod=128):
    """
    Estimate T_eff using the FlucDis-SGD gradient-difference method.

    Two independent mini-batches are passed through separate copies of the
    model and their per-parameter gradients are compared.

    NOTE ON DENOMINATOR: The denominator is ``2 * batch_size * batch1_size``
    where ``batch1_size = x1_enc.shape[0]`` (actual samples in batch1).
    In practice both equal the configured batch size, so the effective
    denominator is ``2 * batch_size^2``. This formulation matched the
    experimental setup used to generate the paper figures and is kept
    unchanged for reproducibility.
    """
    x1, y1 = batch1
    x2, y2 = batch2
    x1, y1 = x1.to(device), y1.to(device)
    x2, y2 = x2.to(device), y2.to(device)

    if task == "modular_add":
        if isinstance(model, TinyMLP):
            x1_enc = torch.cat(
                [F.one_hot(x1[:, 0], num_classes=num_classes_mod),
                 F.one_hot(x1[:, 1], num_classes=num_classes_mod)], dim=1
            ).float()
            x2_enc = torch.cat(
                [F.one_hot(x2[:, 0], num_classes=num_classes_mod),
                 F.one_hot(x2[:, 1], num_classes=num_classes_mod)], dim=1
            ).float()
        else:
            x1_enc, x2_enc = x1, x2
    else:
        x1_enc, x2_enc = x1, x2

    def _batch_grad(x_enc, y):
        rep = copy.deepcopy(model).to(device)
        rep.train()
        rep.zero_grad()
        criterion(rep(x_enc), y).backward()
        return torch.cat([p.grad.detach().view(-1) for p in rep.parameters()])

    grad1 = _batch_grad(x1_enc, y1)
    grad2 = _batch_grad(x2_enc, y2)

    diff = torch.mean((grad1 - grad2) ** 2).item()
    T_eff = lr * diff / (2 * batch_size * x1_enc.shape[0])
    return T_eff


# ---------------------------------------------------------------------------
# Geometry checkpoint
# ---------------------------------------------------------------------------

def save_geometry_checkpoint(model, criterion, X_sample, Y_sample, device, outdir, suffix):
    """Save top-5 Hessian eigenvalues and participation ratio to an .npz file."""
    os.makedirs(outdir, exist_ok=True)
    n_sample = min(256, len(X_sample))
    X_sub = X_sample[:n_sample].to(device)
    Y_sub = Y_sample[:n_sample].to(device)

    try:
        hess_top5 = lanczos_top_k(model, criterion, X_sub, Y_sub, k=5, n_iter=50)
    except Exception:
        hess_top5 = [float('nan')] * 5

    try:
        pr = participation_ratio_from_model(
            model, X_sub, layer_names=['fc2', 'fc1', 'encoder', 'pool']
        )
    except Exception:
        pr = float('nan')

    np.savez(
        os.path.join(outdir, f'geometry_{suffix}.npz'),
        hess_top5=hess_top5,
        participation_ratio=pr,
        step=suffix,
    )
    print(f"  [geometry] saved checkpoint: {suffix}")


# ---------------------------------------------------------------------------
# Evaluation helper (avoids duplicating the batched eval loop)
# ---------------------------------------------------------------------------

def _eval_metrics(model, criterion, X_eval, Y_eval, canonical_logits, device,
                  args, batch_size=256):
    """
    Run full-domain evaluation and return all logged metrics.

    Returns a dict with keys: C_norm, C_PB, m, q_logit, q_ent, test_err,
    hess_top, pr.
    """
    model.eval()
    with torch.no_grad():
        logits_chunks = []
        for i in range(0, len(X_eval), batch_size):
            logits_chunks.append(model(X_eval[i:i + batch_size].to(device)).cpu())
        logits_eval = torch.cat(logits_chunks, dim=0)

        C_norm = compute_C_norm(model)
        C_PB = compute_C_PB(model, sigma_p=args.sigma_p, sigma_q=args.sigma_q)
        m = compute_alignment(logits_eval, canonical_logits) if canonical_logits is not None else 0.0
        q_logit, q_ent = compute_precision(logits_eval)
        test_err = evaluate_test_error(model, X_eval, Y_eval)

    try:
        hess_top5 = lanczos_top_k(
            model, criterion,
            X_eval[:256].to(device), Y_eval[:256].to(device), k=5
        )
        hess_top = hess_top5[0]
    except Exception:
        hess_top = float('nan')

    try:
        pr = participation_ratio_from_model(model, X_eval[:512].to(device))
    except Exception:
        pr = float('nan')

    return dict(C_norm=C_norm, C_PB=C_PB, m=m,
                q_logit=q_logit, q_ent=q_ent,
                test_err=test_err, hess_top=hess_top, pr=pr)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"log_seed{args.seed}.csv")
    checkpoint_path = os.path.join(args.outdir, "checkpoint.pt")

    # ---- Build dataset / eval domain ----
    train_loader, _ = make_dataloaders(args.task, args.n, args.batch,
                                       active_bits=args.active_bits)

    if args.task == "modular_add":
        full_ds = ModularAdditionDataset(n_bits=7, n_samples=None)
        X_list = np.array([x for x, _ in full_ds], dtype=np.int64)
        Y_list = np.array([y for _, y in full_ds], dtype=np.int64)
        X_eval_raw = torch.from_numpy(X_list)
        Y_eval = torch.from_numpy(Y_list)

        a_eval = F.one_hot(X_eval_raw[:, 0], num_classes=128)
        b_eval = F.one_hot(X_eval_raw[:, 1], num_classes=128)
        X_eval_mlp = torch.cat([a_eval, b_eval], dim=1).float()
        canonical_logits = F.one_hot(Y_eval, num_classes=2 ** 7).float()
        num_classes = 2 ** 7

        if args.model == "tiny_mlp":
            X_eval, input_dim = X_eval_mlp, 256
        else:
            X_eval, input_dim = X_eval_raw, 2

    elif args.task == "sparse_parity":
        if args.model != "tiny_mlp":
            raise ValueError("Sparse parity only supports tiny_mlp.")
        active_bits = args.active_bits
        X_eval, Y_eval = generate_full_sparse_parity_domain(active_bits, total_bits=16)
        canonical_logits = canonical_sparse_parity_logits(X_eval, active_bits)
        input_dim = 16
        num_classes = 2
    else:
        raise ValueError(f"Unknown task: {args.task!r}")

    # ---- Build model ----
    if args.model == "tiny_mlp":
        model = TinyMLP(input_dim=input_dim, hidden=args.hidden, num_classes=num_classes)
    elif args.model == "tiny_transformer":
        if args.task != "modular_add":
            raise NotImplementedError("Transformer only supported for modular_add.")
        model = TinyTransformer(
            vocab_size=128, emb=args.emb, nhead=2, nlayers=2, num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model: {args.model!r}")
    model.to(device)

    # ---- Resume from checkpoint ----
    start_step = 0
    if args.resume and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt['step']
        print(f"Resumed from {args.resume_from!r} at step {start_step}.")
    elif args.resume:
        print(f"Checkpoint {args.resume_from!r} not found — starting from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    # ---- CSV header ----
    header = ["step", "time", "train_loss",
              "C_norm", "C_PB", "m", "q_logit", "q_ent",
              "test_err", "hess_top", "PR", "T_eff_proxy"]
    if not args.resume or not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    step = start_step
    start_time = time.time()
    has_grokked = False

    # ---- Step-0 snapshot ----
    if step == 0:
        metrics = _eval_metrics(model, criterion, X_eval, Y_eval,
                                canonical_logits, device, args)
        elapsed = time.time() - start_time
        row = [step, f"{elapsed:.1f}", float('nan'),
               metrics['C_norm'], metrics['C_PB'], metrics['m'],
               metrics['q_logit'], metrics['q_ent'], metrics['test_err'],
               metrics['hess_top'], metrics['pr'], 0.0]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[step {step}] test_err={metrics['test_err']:.3f} (initial)")
        save_geometry_checkpoint(model, criterion, X_eval, Y_eval, device,
                                 args.outdir, 'pre')

    # ---- Training loop ----
    data_iter = iter(train_loader)
    while step < args.max_steps:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            if args.task == "modular_add":
                xs, ys = batch
                xs, ys = xs.to(device), ys.to(device)
                if args.model == "tiny_mlp":
                    inp = torch.cat(
                        [F.one_hot(xs[:, 0], num_classes=128),
                         F.one_hot(xs[:, 1], num_classes=128)], dim=1
                    ).float()
                    logits = model(inp)
                else:
                    logits = model(xs)
            else:
                xs, ys = batch
                xs, ys = xs.to(device), ys.to(device)
                logits = model(xs)

            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.log_interval == 0:
                # Sample a second mini-batch for T_eff estimation.
                try:
                    batch2 = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch2 = next(data_iter)

                T_eff_proxy = compute_teff_flucdis(
                    model, criterion, batch, batch2, device,
                    args.lr, args.batch, args.task, num_classes_mod=128
                )

                metrics = _eval_metrics(model, criterion, X_eval, Y_eval,
                                        canonical_logits, device, args)
                elapsed = time.time() - start_time
                row = [step, f"{elapsed:.1f}", float(loss.item()),
                       metrics['C_norm'], metrics['C_PB'], metrics['m'],
                       metrics['q_logit'], metrics['q_ent'], metrics['test_err'],
                       metrics['hess_top'], metrics['pr'], T_eff_proxy]
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                print(f"[step {step}] loss={loss.item():.4f} "
                      f"test_err={metrics['test_err']:.3f} "
                      f"T_eff={T_eff_proxy:.3e}")

                if not has_grokked and metrics['test_err'] < args.grok_threshold:
                    save_geometry_checkpoint(model, criterion, X_eval, Y_eval,
                                             device, args.outdir, 'at')
                    has_grokked = True

            if step >= args.max_steps:
                break

    # ---- Final checkpoint ----
    save_geometry_checkpoint(model, criterion, X_eval, Y_eval, device,
                             args.outdir, 'post')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, checkpoint_path)
    print(f"Training complete. Log: {csv_path}  Checkpoint: {checkpoint_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model and log grokking metrics."
    )
    parser.add_argument("--task",   type=str,   default="modular_add",
                        choices=["modular_add", "sparse_parity"])
    parser.add_argument("--model",  type=str,   default="tiny_mlp",
                        choices=["tiny_mlp", "tiny_transformer"])
    parser.add_argument("--n",      type=int,   default=500,
                        help="Training-set size.")
    parser.add_argument("--batch",  type=int,   default=8,
                        help="Mini-batch size.")
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--wd",     type=float, default=1e-5,
                        help="Weight decay (λ).")
    parser.add_argument("--sigma_p",type=float, default=1.0)
    parser.add_argument("--sigma_q",type=float, default=1e-5)
    parser.add_argument("--hidden", type=int,   default=256,
                        help="Hidden width for TinyMLP.")
    parser.add_argument("--emb",    type=int,   default=32,
                        help="Embedding dimension for TinyTransformer.")
    parser.add_argument("--max_steps",    type=int,   default=20000)
    parser.add_argument("--log_interval", type=int,   default=100)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--outdir",       type=str,   default="runs")
    parser.add_argument("--grok_threshold", type=float, default=0.1)
    parser.add_argument("--active_bits",  type=int,   default=3,
                        help="Active bits for sparse parity.")
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from checkpoint.")
    parser.add_argument("--resume_from",  type=str,   default="checkpoint.pt",
                        help="Path to checkpoint file.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
