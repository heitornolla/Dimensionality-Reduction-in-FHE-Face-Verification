# Dimensionality-Reduction-in-FHE-Face-Verification

## Running Docker Image

The image offers GPU support. You build the image by executing:

```bash
docker build -t fhe-dim-reduction .
```

And run it with:
```bash
docker run -it --rm --gpus all -v $(pwd):/app fhe-dim-reduction
```

Please, ensure your NVIDIA drivers are updated and Docker is installed.
