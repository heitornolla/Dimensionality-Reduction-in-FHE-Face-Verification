"""
Full experiment: FHE face verification + Autoencoder dimensionality reduction sweep.
Runs multiple AE target dimensions and logs results to CSV.

This is a non-linear, unsupervised alternative to PCA.
"""

import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from baseline_verification import (
    set_deterministic, find_optimal_threshold, get_device
)
from fhe_baseline import (
    get_baseline_embeddings, setup_fhe_context, fhe_distance, 
)


class Autoencoder(nn.Module):
    """
    A simple non-linear Autoencoder.

    Due to the way it is structured, 
    we evaluate dims from 128 to 4.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        
        # 512 -> 256 -> 128 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # latent_dim -> 128 -> 256 -> 512
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(
    data_np: np.ndarray, 
    input_dim: int, 
    latent_dim: int, 
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 128
) -> nn.Module:
    """
    Trains a new Autoencoder on the provided data and returns the trained encoder.
    """
    data_tensor = torch.tensor(data_np, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = Autoencoder(input_dim, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            inputs = batch[0].to(device)
            
            reconstructed = model(inputs)
            
            loss = criterion(reconstructed, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) == epochs:
            print(f"  AE training complete. Final MSE loss: {loss.item():.6f}")

    return model.encoder.eval()


def main(csv_path: str):
    set_deterministic(42)
    device = get_device()
    
    # Get the 6,000 test pairs and labels.
    # emb1 and emb2 are PyTorch Tensors.
    labels, emb1_tensor, emb2_tensor = get_baseline_embeddings()
    orig_dim = emb1_tensor.shape[1] # 512

    emb1_np = emb1_tensor.numpy()
    emb2_np = emb2_tensor.numpy()
    
    # For a just comparison, we train our AE on the same data PCA was fit on.
    all_embs_np = np.vstack((emb1_np, emb2_np))
    
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

    dims_to_test = [128, 64, 32, 16, 8, 4]
    print(f"\nRunning for dimensions: {dims_to_test}")

    for target_dim in dims_to_test:
        print(f"\nAutoencoder {orig_dim} -> {target_dim}")
        
        # Similar to PCA logic, train a new AE for this target dimension
        encoder = train_autoencoder(all_embs_np, orig_dim, target_dim, device)
        
        # Use the trained encoder to reduce our 6,000 pairs
        with torch.no_grad():
            emb1_reduced_t = encoder(emb1_tensor.to(device))
            emb2_reduced_t = encoder(emb2_tensor.to(device))
            
        emb1_reduced = emb1_reduced_t.cpu().numpy()
        emb2_reduced = emb2_reduced_t.cpu().numpy()
        
        emb_dim = emb1_reduced.shape[1]

        print("  Encrypting embeddings")
        ct_db = [cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e)) for e in emb1_reduced]
        ct_probe = [cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e)) for e in emb2_reduced]

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

        print(f"  FHE AE Matching Results for size {target_dim}:")
        print(f"  Average matching time per pair: {avg_time_ms:.3f} ms")
        print(f"  Accuracy of {acc*100:.2f}%")
        print(f"  Optimal threshold of {opt_thresh:.6f}")
        print(f"  Results saved to {csv_path}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.abspath('.'), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "fhe_autoencoder_results.csv")
        
    main(output_csv)