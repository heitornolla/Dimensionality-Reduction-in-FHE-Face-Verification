"""
Face verification with FHE (realistic timing version).

We pre-encrypted embeddings, as we assume that in 
a real scenario, they would be encrypted and stored on a server.
We also pre-encrypt the probe embedding, as we assume a server
would recieve it already encrypted.

We measure the sum of homomorphic matching time and decryption time,
as we consider score decryption an integral part of the matching process.

We DO NOT include any encryption or setup time.
"""

import os
import random
import time
import numpy as np
import torch
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from baseline_verification import (
    get_device, get_model, get_transform,
    embeddings_for_image_batch, find_optimal_threshold, set_deterministic
)


from openfhe import CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature


def setup_fhe_context():
    """
    CKKS setup.
    Returns (cc, keys, fhe_batch_size).
    """
    # Secure polynomial modulus degree (most papers use 8192 or 16384)
    fhe_batch_size = 16384  # ensures >= 128-bit security

    mult_depth = 5
    scale_mod_size = 50        # higher precision per level

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(mult_depth)
    params.SetScalingModSize(scale_mod_size)
    params.SetBatchSize(fhe_batch_size)

    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)
    cc.Enable(PKESchemeFeature.ADVANCEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalSumKeyGen(keys.secretKey)

    return cc, keys


def fhe_distance(cc, ct_a, ct_b, sum_slots):
    """
    Homomorphic squared Euclidean distance between two ciphertexts.
    """
    ct_diff = cc.EvalSub(ct_a, ct_b)
    ct_sq = cc.EvalSquare(ct_diff)
    ct_sum = cc.EvalSum(ct_sq, sum_slots)
    return ct_sum


def get_baseline_embeddings():
    device = get_device()
    model = get_model(device)
    transform = get_transform(160)

    print("Loading LFW pairs")
    lfw = fetch_lfw_pairs(subset="test", color=True, resize=1.0)
    pairs = lfw.pairs
    labels = lfw.target.astype(int)

    print("Computing embeddings")
    imgs = [p[i] for p in pairs for i in (0, 1)]
    all_embs = embeddings_for_image_batch(model, device, transform, imgs, batch_size=128)
    emb1 = all_embs[0::2]
    emb2 = all_embs[1::2]

    return labels, emb1, emb2

def main():
    set_deterministic(42)
    labels, emb1, emb2 = get_baseline_embeddings()

    emb_dim = emb1.shape[1] # 512

    print("Setting FHE Context")
    cc, keys = setup_fhe_context()

    # Pre-encrypt all embeddings (database and probes)
    print("Encrypting embeddings")
    ct_db = [cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e.numpy())) for e in emb1]
    ct_probe = [cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(e.numpy())) for e in emb2]

    # We time encrypted matching + decryption
    print("Running encrypted matching")
    distances = []
    total_time = 0.0

    for i in tqdm(range(len(labels))):
        t0 = time.perf_counter()
        ct_res = fhe_distance(cc, ct_db[i], ct_probe[i], sum_slots=emb_dim)
        pt_res = cc.Decrypt(keys.secretKey, ct_res)
        total_time += time.perf_counter() - t0

        val = float(pt_res.GetRealPackedValue()[0])
        distances.append(val)

    avg_time_ms = (total_time / len(labels)) * 1000

    # Same accuracy evaluation as baseline
    opt_thresh = find_optimal_threshold(labels, np.array(distances))
    preds = (np.array(distances) <= opt_thresh).astype(int)
    acc = accuracy_score(labels, preds)

    print("\nFHE Matching Results:")
    print(f"  Average matching time per pair: {avg_time_ms:.6f}")
    print(f"  Accuracy of {acc * 100:.2f}%")
    print(f"  Optimal threshold of {opt_thresh:.6f}")


if __name__ == "__main__":
    set_deterministic(42)
    main()
