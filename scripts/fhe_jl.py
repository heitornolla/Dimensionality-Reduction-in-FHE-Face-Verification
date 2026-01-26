"""
Author: Heitor Pavani Nolla
All rights reserved

FHE face verification + JL-Based Structured Projection (Hadamard) dimensionality reduction sweep.
Uses Fast Walsh-Hadamard Transform (FWHT) for efficient random projection.
"""

import os
import csv
import time
import numpy as np
from tqdm import tqdm
from scipy.linalg import hadamard

from baseline_verification import get_metrics, set_deterministic

from fhe_baseline import (
    get_test_embeddings,
    setup_fhe_context,
    fhe_distance,
)


def hadamard_projection(
    X: np.ndarray, n_components: int, random_state: int = 42
) -> np.ndarray:
    """
    Structured JL projection using random sign flipping + Hadamard transform + column subsampling.
    Args:
        X: (N, D) input matrix.
        n_components: number of output dimensions.
        random_state: seed for reproducibility.
    Returns:
        Projected data (N, n_components).
    Notes:
        D (original dimension) must be a power of 2 for standard Hadamard.
        If not, we pad X with zeros up to next power of 2.
    """
    rng = np.random.default_rng(random_state)
    N, D = X.shape

    # Pad to next power of 2
    next_pow2 = 1 << (D - 1).bit_length()
    if next_pow2 != D:
        X_padded = np.zeros((N, next_pow2), dtype=np.float64)
        X_padded[:, :D] = X
        D = next_pow2
    else:
        X_padded = X

    # Random sign flipping
    signs = rng.choice([-1.0, 1.0], size=D)
    X_signed = X_padded * signs

    # Hadamard transform (FWHT)
    # We'll apply it via scipy's hadamard matrix for moderate D; for very large D use iterative FWHT.
    H = hadamard(D, dtype=np.float64)
    X_h = (X_signed @ H) / np.sqrt(D)

    # Column subsampling
    idx = rng.choice(D, size=n_components, replace=False)
    X_proj = X_h[:, idx]

    return X_proj.astype(np.float32)


def main(csv_path: str, seed=42):
    set_deterministic(seed)
    labels, emb1, emb2 = get_test_embeddings()

    orig_dim = emb1.shape[1]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = [
        "dimension",
        "avg_time_ms",
        "accuracy(%)",
        "AUC",
        "EER(%)",
        "FAR(%)",
        "FRR(%)",
        "threshold",
    ]

    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    dims_to_test = [512, 256, 128, 64, 32, 16, 8, 4]
    print(f"\nRunning JL-based Hadamard projections for dimensions: {dims_to_test}")

    for target_dim in dims_to_test:
        print(f"\nHadamard JL projection {orig_dim} → {target_dim}")
        emb1_reduced = hadamard_projection(emb1, target_dim, random_state=42)
        emb2_reduced = hadamard_projection(emb2, target_dim, random_state=42)

        emb_dim = emb1_reduced.shape[1]

        cc, keys = setup_fhe_context(target_dim)
        
        ct_db = [
            cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e)) 
            for e in tqdm(emb1_reduced)
        ]
        ct_probe = [
            cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e))
            for e in tqdm(emb2_reduced)
        ]

        # Encrypted matching
        distances = []
        total_time = 0.0
        for i in tqdm(range(len(labels)), leave=False):
            t0 = time.perf_counter()
            ct_res = fhe_distance(cc, ct_db[i], ct_probe[i], sum_slots=emb_dim)
            pt_res = cc.Decrypt(keys.secretKey, ct_res)
            total_time += time.perf_counter() - t0
            distances.append(float(pt_res.GetRealPackedValue()[0]))

        avg_time_ms = (total_time / len(labels)) * 1000

        metrics = get_metrics(labels, np.array(distances))

        row = [
            target_dim,
            f"{avg_time_ms:.3f}",
            f"{metrics['accuracy']:.2f}",
            f"{metrics['auc']:.4f}",
            f"{metrics['eer']:.2f}",
            f"{metrics['far']:.2f}",
            f"{metrics['frr']:.2f}",
            f"{metrics['threshold']:.6f}",
        ]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"  Results saved to {csv_path}")


if __name__ == "__main__":
    output_csv = f"{os.path.abspath('./')}/results/fhe_jl_results.csv"
    main(output_csv)
