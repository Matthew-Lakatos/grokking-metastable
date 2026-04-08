import torch
import numpy as np

def generate_full_parity_domain(n_bits=16):
    """Generate all 2^n_bits bitstrings and their XOR parity labels."""
    num_samples = 2 ** n_bits
    X = np.zeros((num_samples, n_bits), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)
    for i in range(num_samples):
        bits = [(i >> b) & 1 for b in range(n_bits)]
        X[i] = bits
        y[i] = sum(bits) % 2
    return torch.from_numpy(X), torch.from_numpy(y)

def canonical_parity_logits(X, n_bits=16):
    """Canonical logits: one‑hot of true parity (2 classes)."""
    _, y = generate_full_parity_domain(n_bits)
    return torch.nn.functional.one_hot(y, num_classes=2).float()
