# Comparison of Unsupervised Dimensionality Reduction Methods for Fully Homomorphic Encrypted Facial Verification

This repository contains codes for the paper:

**“Comparison of Unsupervised Dimensionality Reduction Methods for Fully Homomorphic Encrypted Facial Verification”**, accepted at IWBF 2026.

We systematically evaluate how unsupervised dimensionality reduction techniques impact **accuracy** and **computational efficiency** in a **Fully Homomorphic Encryption (FHE)**-based face verification pipeline.

---

## Overview

FHE enables biometric matching directly on encrypted data, providing end-to-end template protection. However, FHE introduces substantial computational overhead. We show that compressing face embeddings _before encryption_ can significantly reduce homomorphic computation time while preserving verification accuracy.

---

## Running the Codes

### Docker

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

### Local Python Install

If you choose to not use Docker, you can just create a virtual environment and download the required packages from the `requirements.txt` file. From there, merely run the scripts in the `/scripts` folder. You may run our exact pipeline by running `reproduce_paper_results.py`, but beware that the experiments may take a long time to finish.

We recommend `Python >3.12` for these codes. Specifically, we used version `3.12.7` and [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) as a package manager  


## Contact Information

For questions about this work, feel free to contact Heitor Nolla at [LinkedIn](https://www.linkedin.com/in/heitor-nolla/) or email: h173233@dac.unicamp.br


## Contact Information

For questions about this work, feel free to contact Heitor Nolla at [LinkedIn](https://www.linkedin.com/in/heitor-nolla/) or email: h173233@dac.unicamp.br
