# Comparison of Unsupervised Dimensionality Reduction Methods for Fully Homomorphic Encrypted Facial Verification

This repository contains codes for the paper:

**“Comparison of Unsupervised Dimensionality Reduction Methods for Fully Homomorphic Encrypted Facial Verification”**, submitted to IWBF 2026.

The goal of this project is to systematically evaluate how unsupervised dimensionality reduction techniques impact **accuracy** and **computational efficiency** in a **Fully Homomorphic Encryption (FHE)**-based face verification pipeline.

---

## Overview

FHE enables biometric matching directly on encrypted data, providing end-to-end template protection. However, FHE introduces substantial computational overhead.

This repository explores **dimensionality reduction as a practical, encryption-agnostic optimization**, showing that compressing face embeddings _before encryption_ can significantly reduce homomorphic computation time while preserving verification accuracy.

### Key Findings

- Reducing FaceNet embeddings from **512-D to 32-D** yields:
  - ~**1.5× speedup** in homomorphic matching
  - **No measurable loss in accuracy or EER** (with data-driven methods)
- Data-driven methods (PCA, RSVD, Autoencoders) are more robust at aggressive compression
- Training-free methods (Random Projections, JL-Hadamard) perform competitively at moderate dimensions

---

## Evaluated Techniques

The following **unsupervised dimensionality reduction methods** are implemented and compared:

- PCA
- RSVD
- Autoencoder (AE)
- Gaussian Random Projection (GRP)
- Sparse Random Projection (SRP)
- JL-Hadamard

---

## Running the Codes

We provide a Docker image with Pytorch and GPU support. You may build the image by executing:

```bash
docker build -t fhe-dim-reduction .
```

And run it with:

```bash
docker run -it --rm --gpus all -v $(pwd):/app fhe-dim-reduction
```

Please, ensure your NVIDIA drivers are updated and Docker is installed.
The image runs on CUDA 13.0. If compatibility issues arise, you may find versions which match your drivers [here](https://hub.docker.com/r/pytorch/pytorch/).

From inside of the image, all of the experiments are located in the `scripts` folder. You may run them by executing:

```bash
python scripts/reproduce_paper_results.py
```

## Contact Information

For questions about this work, feel free to contact Heitor Nolla at [LinkedIn](https://www.linkedin.com/in/heitor-nolla/) or email: h173233@dac.unicamp.br
