"""
Author: Heitor Pavani Nolla
All rights reserved

Baseline face verification code.
"""

import os
import random 
from typing import List, Tuple
import time

import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_curve, accuracy_score
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
            transforms.ToTensor(),  # yields floats in [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def to_pil_safe(arr: np.ndarray) -> Image.Image:
    """
    Convert a numpy image array to a PIL Image.
    """
    if arr.dtype == np.uint8:
        return Image.fromarray(arr)
    # floats
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
    Compute embeddings for a list of numpy images (H,W,3).
    Returns a torch.Tensor of shape (len(images), embedding_dim).
    """
    all_embs: List[Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            sub = images[i : i + batch_size]
            tensors = []
            for arr in sub:
                pil = to_pil_safe(arr)
                tensors.append(transform(pil))
            batch_tensor = torch.stack(tensors, dim=0).to(device)
            emb = model(batch_tensor)  # (B, 512)
            all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def squared_euclidean(a: Tensor, b: Tensor) -> np.ndarray:
    """Compute squared euclidean distance per-pair between two tensors (same length)."""
    # a, b are torch tensors on CPU
    diff = a - b
    dist = diff.pow(2).sum(dim=1).numpy()
    return dist


def find_optimal_threshold(labels: np.ndarray, dists: np.ndarray) -> float:
    """
    Given binary labels (1 = same) and distances (lower = more similar),
    find threshold that maximizes (tpr - fpr) on ROC curve.
    Returns distance threshold (not inverted).
    """
    inv = -dists  # higher = better
    fpr, tpr, thresholds = roc_curve(labels, inv, pos_label=1)
    opt_idx = np.argmax(tpr - fpr)
    opt_threshold_inv = thresholds[opt_idx]
    return -opt_threshold_inv


def evaluate_lfw_pairs(
    model: InceptionResnetV1,
    device: Device,
    transform: transforms.Compose,
    subset: str = "test",
    batch_size: int = 64,
) -> Tuple[float, float, float]:
    """
    Evaluate LFW pairs (subset 'test' default).
    Returns (optimal_threshold, accuracy, avg_match_time_ms)
    """
    print("Loading LFW pairs...")
    lfw = fetch_lfw_pairs(subset=subset, color=True, resize=1.0)
    pairs = lfw.pairs  # shape (N, 2, H, W, C)
    labels = lfw.target.astype(int)
    num_pairs = len(labels)

    # Build flattened image list: [img1_pair0, img2_pair0, img1_pair1, img2_pair1, ...]
    flat_images = []
    for p in pairs:
        flat_images.append(p[0])
        flat_images.append(p[1])

    # Compute embeddings for all images (this computes each image once)
    all_embs = embeddings_for_image_batch(model, device, transform, flat_images, batch_size=batch_size)
    # all_embs shape: (2*num_pairs, 512)

    # Split back into per-pair embeddings
    emb1 = all_embs[0::2]
    emb2 = all_embs[1::2]

    # measure match time for distance computation
    start_time = time.time()
    dists = squared_euclidean(emb1, emb2)
    total_time = time.time() - start_time

    # find threshold
    optimal_threshold = find_optimal_threshold(labels, dists)
    predictions = (dists <= optimal_threshold).astype(int)
    acc = accuracy_score(labels, predictions)
    avg_match_time_ms = (total_time / num_pairs) * 1000.0

    return optimal_threshold, float(acc), float(avg_match_time_ms)


def main():
    set_deterministic(42)
    device = get_device()
    print(f"Running on device: {device}")
    model = get_model(device)
    transform = get_transform(160)

    thresh, acc, avg_time_ms = evaluate_lfw_pairs(model, device, transform, subset="test", batch_size=128)

    print("\nBaseline Results")
    print(f"  Average matching time per pair: {avg_time_ms:.6f}")
    print(f"  Accuracy of {acc * 100:.2f}%")
    print(f"  Optimal threshold of {thresh:.6f}")


if __name__ == "__main__":
    main()
