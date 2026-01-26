"""
Author: Heitor Pavani Nolla
All rights reserved

Face verification with FHE.
"""

import time
import numpy as np
from sklearn.datasets import fetch_lfw_pairs
from tqdm import tqdm

from baseline_verification import (
    cross_validate_lfw,
    get_device,
    get_model,
    get_transform,
    embeddings_for_image_batch,
    set_deterministic,
)


from openfhe import (
    CCParamsCKKSRNS,
    CryptoContext,
    GenCryptoContext,
    KeyPair,
    PKESchemeFeature,
    SecurityLevel
)
from typing import Tuple


def setup_fhe_context(target_dim: int = 512) -> Tuple[CryptoContext, KeyPair]:
    """
    CKKS setup optimized for a specific vector dimension.
    """

    fhe_batch_size = max(8, target_dim) 

    mult_depth = 2 
    scale_mod_size = 50

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(scale_mod_size)
    params.SetBatchSize(fhe_batch_size)
    
    # Explicitly request 128-bit security to ensure fair comparison
    params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    
    if target_dim > 1:
        rotations = []
        cur = 1
        while cur < target_dim:
            rotations.append(cur)
            cur *= 2
        cc.EvalRotateKeyGen(keys.secretKey, rotations)

    return cc, keys


def fhe_distance(cc: CryptoContext, ct_a, ct_b, sum_slots: int):
    """
    Homomorphic squared Euclidean distance.
    """
    ct_diff = cc.EvalSub(ct_a, ct_b)
    ct_sq = cc.EvalSquare(ct_diff)
    ct_sum = cc.EvalSum(ct_sq, sum_slots)
    return ct_sum


def get_training_data(model=None, device=None, transform=None) -> np.ndarray:
    """
    Loads the LFW "train" subset and returns all embeddings as a single
    NumPy array for training/fitting unsupervised DR models.
    """

    print("Loading LFW 'train' subset for training DR models")
    if device is None:
        device = get_device()
    if model is None:
        model = get_model(device)
    if transform is None:
        transform = get_transform(160)

    lfw_train = fetch_lfw_pairs(subset="train", color=True, resize=1.0)
    pairs = lfw_train.pairs

    flat_images = []
    for p in pairs:
        flat_images.append(p[0])
        flat_images.append(p[1])

    # We ignore the pairing and labels; we just want the raw data distribution
    all_embs_tensor = embeddings_for_image_batch(
        model, device, transform, flat_images, batch_size=128
    )

    print(f"Loaded {len(all_embs_tensor)} training embeddings.")
    return all_embs_tensor.numpy()


def get_test_embeddings(
    model=None, device=None, transform=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the LFW "test" subset and returns the labels and separated
    embedding pairs as NumPy arrays for evaluation.

    Returns:
        labels (np.ndarray): (6000,) array of 0s and 1s
        emb1_np (np.ndarray): (6000, 512) array of "left" embeddings
        emb2_np (np.ndarray): (6000, 512) array of "right" embeddings
    """

    if device is None:
        device = get_device()
    if model is None:
        model = get_model(device)
    if transform is None:
        transform = get_transform(160)

    lfw_test = fetch_lfw_pairs(subset="test", color=True, resize=1.0)
    pairs = lfw_test.pairs
    labels = lfw_test.target.astype(int)
    num_pairs = len(labels)

    flat_images = []
    for p in pairs:
        flat_images.append(p[0])
        flat_images.append(p[1])

    all_embs_tensor = embeddings_for_image_batch(
        model, device, transform, flat_images, batch_size=128
    )

    emb1_np = all_embs_tensor[0::2].numpy()
    emb2_np = all_embs_tensor[1::2].numpy()

    print(f"Loaded {num_pairs} test pairs.")
    return labels, emb1_np, emb2_np


def main(csv_path: str = "results/fhe_baseline.csv", seed=42):
    set_deterministic(seed)
    device = get_device()
    model = get_model(device)
    transform = get_transform(160)

    labels, emb1_np, emb2_np = get_test_embeddings(model, device, transform)

    cc, keys = setup_fhe_context()

    # Pre-encrypt all embeddings (database and probes)
    ct_db = [
        cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e)) for e in tqdm(emb1_np)
    ]
    ct_probe = [
        cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e)) for e in tqdm(emb2_np)
    ]

    distances = []
    total_time = 0.0
    for i in tqdm(range(len(labels))):
        t0 = time.perf_counter()
        ct_res = fhe_distance(cc, ct_db[i], ct_probe[i], sum_slots=512)
        pt_res = cc.Decrypt(keys.secretKey, ct_res)
        total_time += time.perf_counter() - t0
        distances.append(float(pt_res.GetRealPackedValue()[0]))

    accuracies = cross_validate_lfw(labels, np.array(distances), n_folds=10)
    
    mean_acc = np.mean(accuracies) * 100
    std_acc = np.std(accuracies) * 100
    avg_time_ms = (total_time / len(labels)) * 1000

    print("\nFHE Baseline Results:")
    print(f"  Avg matching time: {avg_time_ms:.3f} ms")
    print(f"  Accuracy of {mean_acc:.2f}% (+/- {std_acc:.2f}%)")

if __name__ == "__main__":
    main()
