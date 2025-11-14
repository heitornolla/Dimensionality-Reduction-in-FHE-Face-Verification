"""
Author: Heitor Pavani Nolla
All rights reserved

Full experiment: FHE face verification + Gaussian Random Projection (GRP) dimensionality reduction sweep.
Runs multiple GRP target dimensions and logs results to CSV.
"""

import os
import csv
import time
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from baseline_verification import (
    set_deterministic, find_optimal_threshold
)

from fhe_baseline import (
    get_test_embeddings, setup_fhe_context, fhe_distance,
)


def main(csv_path: str):
    set_deterministic(42)
    labels, emb1, emb2 = get_test_embeddings()

    orig_dim = emb1.shape[1]
    print(f"Original embedding dimension: {orig_dim}")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = [
        "dimension",
        "avg_time_ms",
        "accuracy(%)",
        "threshold"
    ]
    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    print("Setting FHE context")
    cc, keys = setup_fhe_context()

    dims_to_test = [512, 256, 128, 64, 32, 16, 8, 4]
    print(f"\nRunning for dimensions: {dims_to_test}")

    for target_dim in dims_to_test:
        print(f"\nGaussian Random Projection {orig_dim} → {target_dim}")
        grp = GaussianRandomProjection(n_components=target_dim, random_state=42)
        emb1_reduced = grp.fit_transform(emb1)
        emb2_reduced = grp.transform(emb2)

        emb_dim = emb1_reduced.shape[1]

        # Encrypt reduced embeddings
        print("  Encrypting embeddings")
        ct_db = [cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e)) for e in tqdm(emb1_reduced)]
        ct_probe = [cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e)) for e in tqdm(emb2_reduced)]

        # Encrypted matching
        print("  Running encrypted matching")
        distances = []
        total_time = 0.0
        for i in tqdm(range(len(labels)), leave=False):
            t0 = time.perf_counter()
            ct_res = fhe_distance(cc, ct_db[i], ct_probe[i], sum_slots=emb_dim)
            pt_res = cc.Decrypt(keys.secretKey, ct_res)
            total_time += time.perf_counter() - t0
            distances.append(float(pt_res.GetRealPackedValue()[0]))

        avg_time_ms = (total_time / len(labels)) * 1000

        opt_thresh = find_optimal_threshold(labels, np.array(distances))
        preds = (np.array(distances) <= opt_thresh).astype(int)
        acc = accuracy_score(labels, preds)

        row = [
            target_dim,
            f"{avg_time_ms:.3f}",
            f"{acc*100:.2f}",
            f"{opt_thresh:.6f}",
        ]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"  FHE GRP Matching Results for size {target_dim}:")
        print(f"  Average matching time per pair: {avg_time_ms:.3f} ms")
        print(f"  Accuracy of {acc*100:.2f}%")
        print(f"  Optimal threshold of {opt_thresh:.6f}")
        print(f"  Results saved to {csv_path}")


if __name__ == "__main__":
    output_csv = f"{os.path.abspath('./')}/results/fhe_grp_results.csv"
    main(output_csv)
