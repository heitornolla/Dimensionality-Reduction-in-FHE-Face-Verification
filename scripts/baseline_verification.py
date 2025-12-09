"""
Author: Heitor Pavani Nolla
All rights reserved

Baseline face verification code.
"""

import os
import random
from typing import List, Dict
import time

import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from PIL import Image

Device = torch.device


def set_deterministic(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> Device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(device: Device) -> InceptionResnetV1:
    """
    Load pretrained FaceNet (InceptionResnetV1).
    """
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model


def get_transform(image_size: int = 160) -> transforms.Compose:
    """
    Return the transform used for InceptionResnetV1 (expects inputs in [-1, 1]).
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def to_pil_safe(arr: np.ndarray) -> Image.Image:
    """
    Convert a numpy image array to a PIL Image.
    """
    if arr.dtype == np.uint8:
        return Image.fromarray(arr)
    arr_max = float(np.max(arr))
    if arr_max <= 1.0:
        arr_uint8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr_uint8 = arr.clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr_uint8)


def embeddings_for_image_batch(
    model: InceptionResnetV1,
    device: Device,
    transform: transforms.Compose,
    images: List[np.ndarray],
    batch_size: int = 128,
) -> Tensor:
    """
    Compute embeddings for a list of numpy images.
    """
    all_embs: List[Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            sub = images[i : i + batch_size]
            tensors = [transform(to_pil_safe(arr)) for arr in sub]
            batch_tensor = torch.stack(tensors, dim=0).to(device)
            emb = model(batch_tensor)
            all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def squared_euclidean(a: Tensor, b: Tensor) -> np.ndarray:
    """
    Compute squared euclidean distance per-pair (vectorized).
    """
    diff = a - b
    dist = diff.pow(2).sum(dim=1).numpy()
    return dist


def find_optimal_threshold(labels: np.ndarray, dists: np.ndarray) -> float:
    """
    Legacy helper: Find threshold that maximizes (tpr - fpr) on ROC curve.
    Kept for compatibility, but get_metrics is preferred.
    """
    inv = -dists
    fpr, tpr, thresholds = roc_curve(labels, inv, pos_label=1)
    opt_idx = np.argmax(tpr - fpr)
    opt_threshold_inv = thresholds[opt_idx]
    return -opt_threshold_inv


def get_metrics(labels: np.ndarray, dists: np.ndarray) -> Dict[str, float]:
    """
    Calculates Accuracy, AUC, EER, FAR, FRR, and Optimal Threshold.
    """
    # Invert distances (lower distance = higher score)
    scores = -dists

    fprs, tprs, thresholds = roc_curve(labels, scores, pos_label=1)

    auc = roc_auc_score(labels, scores)

    # EER (Equal Error Rate): Point where FAR (FPR) ~= FRR (1-TPR)
    fnrs = 1 - tprs
    eer_idx = np.nanargmin(np.absolute(fnrs - fprs))
    eer = fprs[eer_idx]

    # Optimal Threshold (Max TPR - FPR)
    opt_idx = np.argmax(tprs - fprs)
    optimal_threshold = -thresholds[opt_idx]  # Revert to distance scale

    # Metrics at Optimal Threshold
    # FAR = FPR at optimal threshold
    # FRR = FNR at optimal threshold
    final_far = fprs[opt_idx]
    final_frr = fnrs[opt_idx]

    preds = (dists <= optimal_threshold).astype(int)
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc * 100,
        "auc": auc,
        "eer": eer * 100,
        "far": final_far * 100,
        "frr": final_frr * 100,
        "threshold": optimal_threshold,
    }


def main():
    set_deterministic(42)
    device = get_device()
    print(f"Running on device: {device}")
    model = get_model(device)
    transform = get_transform(160)

    print("Loading LFW pairs...")
    lfw = fetch_lfw_pairs(subset="test", color=True, resize=1.0)
    pairs = lfw.pairs
    labels = lfw.target.astype(int)
    num_pairs = len(labels)

    # Build flattened image list
    flat_images = []
    for p in pairs:
        flat_images.append(p[0])
        flat_images.append(p[1])

    all_embs = embeddings_for_image_batch(
        model, device, transform, flat_images, batch_size=64
    )
    emb1 = all_embs[0::2]
    emb2 = all_embs[1::2]

    # measure match time for distance computation
    start_time = time.time()
    dists = squared_euclidean(emb1, emb2)
    total_time = time.time() - start_time

    metrics = get_metrics(labels, dists)
    avg_match_time_ms = (total_time / num_pairs) * 1000.0

    print(f"Avg Match Time: {avg_match_time_ms:.6f} ms")
    print(f"Accuracy:       {metrics['accuracy']:.2f}%")
    print(f"AUC:            {metrics['auc']:.4f}")
    print(f"EER:            {metrics['eer']:.2f}%")
    print(f"FAR:            {metrics['far']:.2f}%")
    print(f"FRR:            {metrics['frr']:.2f}%")
    print(f"Threshold:      {metrics['threshold']:.6f}")


if __name__ == "__main__":
    main()
