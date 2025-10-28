FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake libssl-dev libffi-dev libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app
CMD ["/bin/bash"]
