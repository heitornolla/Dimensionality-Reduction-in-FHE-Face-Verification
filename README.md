# Comparison of Unsupervised Dimensionality Reduction Methods for Fully Homomorphic Encrypted Facial Verification

This repository contains the code accompanying the paper:

> **Heitor Pavani Nolla and André Leon S. Gradvohl.**  
> *Comparison of Unsupervised Dimensionality Reduction Methods for Fully Homomorphic Encrypted Facial Verification.*  
> In **Proceedings of the 2026 14th International Workshop on Biometrics and Forensics (IWBF)**, pp. 1–6, 2026.  
> DOI: https://doi.org/10.1109/IWBF68042.2026.11558161

If you use this repository in your research, please cite:

```bibtex
@INPROCEEDINGS{11558161,
  author={Nolla, Heitor Pavani and Gradvohl, André Leon S.},
  booktitle={2026 14th International Workshop on Biometrics and Forensics (IWBF)},
  title={Comparison of Unsupervised Dimensionality Reduction Methods for Fully Homomorphic Encrypted Facial Verification},
  year={2026},
  pages={1-6},
  keywords={Biometrics;Dimensionality reduction;Accuracy;Faces;Printing;Timing;Training;Homomorphic encryption;Vectors;Optimization;homomorphic encryption;dimensionality reduction;face verification},
  doi={10.1109/IWBF68042.2026.11558161}
}
```

We systematically evaluate how unsupervised dimensionality reduction techniques impact **accuracy** and **computational efficiency** in a **Fully Homomorphic Encryption (FHE)**-based face verification pipeline.

---

## Overview

FHE enables biometric matching directly on encrypted data, providing end-to-end template protection. However, FHE introduces substantial computational overhead. We show that compressing face embeddings *before encryption* can significantly reduce homomorphic computation time while preserving verification accuracy.

## Running the Code

### Docker

We provide a Docker image with PyTorch and GPU support. You can build the image by executing:

```bash
docker build -t fhe-dim-reduction .
```

Then run it with:

```bash
docker run -it --rm --gpus all -v $(pwd):/app fhe-dim-reduction
```

Please ensure your NVIDIA drivers are up to date and Docker is installed.

The image runs on CUDA 13.0. If compatibility issues arise, you can find PyTorch images matching your CUDA version here:

https://hub.docker.com/r/pytorch/pytorch/

From inside the container, all experiments are located in the `scripts` folder. You can reproduce the paper results by running:

```bash
python scripts/reproduce_paper_results.py
```

### Local Python Installation

If you prefer not to use Docker, create a virtual environment and install the required packages listed in `requirements.txt`.

Then run the scripts in the `scripts` directory. To reproduce the complete experimental pipeline, execute:

```bash
python scripts/reproduce_paper_results.py
```

Note that the experiments may take a considerable amount of time to complete.

We recommend **Python 3.12 or newer**. Our experiments were conducted using **Python 3.12.7** with **Miniconda** as the package manager.

## Contact Information

For questions about this work, feel free to contact **Heitor Nolla**:

- LinkedIn: https://www.linkedin.com/in/heitor-nolla/
- Email: h173233@dac.unicamp.br
