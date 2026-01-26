"""
Author: Heitor Pavani Nolla
All rights reserved

FHE face verification + PCA dimensionality reduction sweep.
Runs multiple PCA target dimensions and logs results to CSV.

- Fits PCA on the train subset.
- Transforms and evaluates on the test subset.
"""

import os
import csv
import time
import numpy as np
from sklearn.decomposition import PCA
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
    get_training_data,
    setup_fhe_context,
    fhe_distance,
)


def main(csv_path: str, seed=42):
    set_deterministic(seed)

    device = get_device()
    model = get_model(device)
    transform = get_transform(160)

    train_data_np = get_training_data(model, device, transform)

    print("Fitting PCA")
    
    labels, emb1_np, emb2_np = get_test_embeddings(model, device, transform)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    header = [
        "dimension",
        "avg_time_ms",
        "mean_accuracy(%)",
        "std_dev(%)",
        "AUC",
        "threshold",
    ]

    write_header = not os.path.exists(csv_path)
    if write_header:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # Test multiple PCA dimensions
    dims_to_test = [512, 256, 128, 64, 32, 16, 8, 4]

    print(f"\nRunning for dimensions: {dims_to_test}")
    for target_dim in dims_to_test:
        print(f"\nPCA to {target_dim}")
        train_reduced = pca.fit_transform(train_data_np)
        
        pca = PCA(n_components=target_dim, random_state=42)
        emb1_reduced = pca.transform(emb1_np)
        emb2_reduced = pca.transform(emb2_np)

        cc, keys = setup_fhe_context(target_dim)
        
        ct_db = [
            cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e)) 
            for e in tqdm(emb1_reduced)
        ]
        ct_probe = [
            cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e))
            for e in tqdm(emb2_reduced)
        ]

        emb_dim = emb1_reduced.shape[1]

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

        accuracies = cross_validate_lfw(labels, np.array(distances), n_folds=10)
        mean_acc = np.mean(accuracies) * 100
        std_acc = np.std(accuracies) * 100
        
        # Additional global metrics for reference
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
    output_dir = os.path.join(os.path.abspath("."), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "fhe_pca_results.csv")
    main(output_csv)
