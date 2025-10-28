# Dimensionality-Reduction-in-FHE-Face-Verification

## Running Docker Image

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
python scripts/baseline_verification.py
```