#!/usr/bin/env python3
"""
run_experiment.py
Minimal experiment runner for grokking diagnostics.
Usage example:
  python run_experiment.py --task modular_add --model tiny_mlp --n 500 --batch 8 --wd 1e-5 --seed 0
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

# Import diagnostics (assumes diagnostics/order_params.py is in PYTHONPATH)
from diagnostics.order_params import (
    compute_C_norm,
    compute_C_PB,
    compute_alignment,
    compute_precision,
    evaluate_test_error,
    lanczos_top_eig,
    participation_ratio_from_activations,
)

# generate datasets
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
            x = np.array([a, b], dtype=np.int64)
            y = (a + b) % self.mod
            self.inputs.append(x)
            self.targets.append(y)
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


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, emb=32, nhead=2, nlayers=2, num_classes=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pool = nn.Linear(emb, num_classes if num_classes is not None else vocab_size)
    def forward(self, x):
        # x: (batch, seq_len) ints
        e = self.emb(x).permute(1,0,2)  # seq_len, batch, emb
        h = self.encoder(e)            # seq_len, batch, emb
        h = h.mean(dim=0)              # batch, emb
        return self.pool(h)

# utils
def make_dataloaders(task, n, batch_size):
    if task == "modular_add":
        ds = ModularAdditionDataset(n_bits=7, n_samples=n)
        # encode inputs as two integers; model expects floats or embeddings
        def collate(batch):
            xs = torch.tensor([b[0] for b in batch], dtype=torch.long)
            ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
            return xs, ys
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
        return loader, ds  # return dataset for full-domain eval
    elif task == "parity":
        ds = ParityDataset(n_bits=16, n_samples=n)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        return loader, ds
    else:
        raise ValueError("Unknown task")

# training loop
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"log_seed{args.seed}.csv")
    # Build data
    train_loader, train_ds = make_dataloaders(args.task, args.n, args.batch)
    # Build model
    if args.model == "tiny_mlp":
        if args.task == "parity":
            input_dim = train_ds.X.shape[1]
            model = TinyMLP(input_dim=input_dim, hidden=args.hidden, num_classes=2)
        else:
            # modular add: represent inputs as two ints -> embed or flatten
            model = TinyMLP(input_dim=2, hidden=args.hidden, num_classes=2**7)
    elif args.model == "tiny_transformer":
        if args.task == "modular_add":
            model = TinyTransformer(vocab_size=2**7, emb=args.emb, num_classes=2**7)
        else:
            raise NotImplementedError("Transformer only implemented for modular_add in skeleton")
    else:
        raise ValueError("Unknown model")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    # Prepare evaluation grid (full domain for small tasks)
    if args.task == "modular_add":
        # enumerate full domain for alignment and test
        full_ds = ModularAdditionDataset(n_bits=7, n_samples=None)
        X_eval = torch.tensor([x for x, _ in full_ds], dtype=torch.long)
        Y_eval = torch.tensor([y for _, y in full_ds], dtype=torch.long)
    else:
        # parity: use train_ds as eval proxy
        X_eval = torch.tensor(train_ds.X, dtype=torch.float32)
        Y_eval = torch.tensor(train_ds.y, dtype=torch.long)
    # canonical logits for modular_add: one-hot teacher
    if args.task == "modular_add":
        canonical_logits = F.one_hot(Y_eval, num_classes=2**7).float()
    else:
        canonical_logits = None

    # CSV header
    header = ["step","time","train_loss","C_norm","C_PB","m","q_logit","q_ent","test_err","hess_top","PR"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    step = 0
    last_log = 0
    start_time = time.time()
    while step < args.max_steps:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            if args.task == "modular_add":
                xs, ys = batch  # xs: (B,2) long
                xs = xs.to(device)
                ys = ys.to(device)
                # For MLP: convert ints to floats (simple) or use embedding in transformer
                if args.model == "tiny_mlp":
                    inp = xs.float()
                    logits = model(inp)
                else:
                    logits = model(xs.to(device))
            else:  # parity
                xs, ys = batch
                xs = xs.to(device)
                ys = ys.to(device)
                logits = model(xs)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.log_interval == 0:
                # compute diagnostics
                model.eval()
                with torch.no_grad():
                    C_norm = compute_C_norm(model).item()
                    C_PB = compute_C_PB(model, sigma_p=args.sigma_p, sigma_q=args.sigma_q).item()
                    # compute logits over X_eval in batches
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
                    test_err = evaluate_test_error(model, X_eval, Y_eval, args)
                    # Hessian top eigenvalue (cheap approx: power iteration on loss Hessian)
                    try:
                        hess_top = lanczos_top_eig(model, criterion, X_eval[:min(256,len(X_eval))].to(device), Y_eval[:min(256,len(Y_eval))].to(device), k=1)
                        hess_top = float(hess_top[0])
                    except Exception as e:
                        hess_top = float("nan")
                    # participation ratio from activations (use a forward hook or recompute features)
                    try:
                        pr = participation_ratio_from_activations(model, X_eval[:min(512,len(X_eval))].to(device))
                    except Exception:
                        pr = float("nan")
                elapsed = time.time() - start_time
                row = [step, f"{elapsed:.1f}", float(loss.item()), C_norm, C_PB, m, q_logit, q_ent, test_err, hess_top, pr]
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f"[step {step}] loss={loss.item():.4f} test_err={test_err:.3f} C_norm={C_norm:.3f} m={m:.3f}")
            if step >= args.max_steps:
                break
        # end epoch
    # end training
    print("Training finished. Logs saved to", csv_path)

# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="modular_add", choices=["modular_add","parity"])
    parser.add_argument("--model", type=str, default="tiny_mlp", choices=["tiny_mlp","tiny_transformer"])
    parser.add_argument("--n", type=int, default=500, help="dataset size (samples)")
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
