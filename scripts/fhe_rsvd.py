"""
Author: Heitor Pavani Nolla
All rights reserved
FHE face verification + RSVD dimensionality reduction sweep.
"""

import os
import csv
import time
import numpy as np
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

from baseline_verification import (
    cross_validate_lfw,
    get_metrics,
    set_deterministic,
    get_device,
    get_model,
    get_transform,
)

from fhe_baseline import (
    get_test_embeddings,
    setup_fhe_context,
    fhe_distance,
)

def main(csv_path: str, seed=42):
    set_deterministic(seed)
    device = get_device()
    model = get_model(device)
    transform = get_transform(160)

    labels, emb1_np, emb2_np = get_test_embeddings(model, device, transform)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    header = [
        "dimension", 
        "avg_time_ms", 
        "mean_accuracy(%)", 
        "std_dev(%)", 
        "AUC", 
        "threshold"
    ]
    
    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    target_dims = [512, 256, 128, 64, 32, 16, 8, 4]

    for target_dim in target_dims:
        print(f"\nEvaluating RSVD dimension: {target_dim}")
        
        U, S, Vt = randomized_svd(emb1_np, n_components=target_dim, random_state=42)
        emb1_reduced = emb1_np @ Vt.T
        emb2_reduced = emb2_np @ Vt.T

        emb_dim = emb1_reduced.shape[1]
        cc, keys = setup_fhe_context(target_dim=emb_dim)

        ct_db = [cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e)) for e in emb1_reduced]
        ct_probe = [cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e)) for e in emb2_reduced]

        distances = []
        total_time = 0.0
        for i in tqdm(range(len(labels)), leave=False):
            t0 = time.perf_counter()
            ct_res = fhe_distance(cc, ct_db[i], ct_probe[i], sum_slots=emb_dim)
            pt_res = cc.Decrypt(keys.secretKey, ct_res)
            total_time += time.perf_counter() - t0
            distances.append(float(pt_res.GetRealPackedValue()[0]))

        accuracies = cross_validate_lfw(labels, np.array(distances), n_folds=10)
        mean_acc = np.mean(accuracies) * 100
        std_acc = np.std(accuracies) * 100
        
        avg_time_ms = (total_time / len(labels)) * 1000
        metrics = get_metrics(labels, np.array(distances))

        row = [
            target_dim,
            f"{avg_time_ms:.3f}",
            f"{mean_acc:.2f}",
            f"{std_acc:.2f}",
            f"{metrics['auc']:.4f}",
            f"{metrics['threshold']:.6f}",
        ]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"  Results saved to {csv_path}")

if __name__ == "__main__":
    main("results/fhe_rsvd.csv")
