#!/usr/bin/env bash
set -euo pipefail

IMAGE="nvcr.io/nvidia/tensorflow:25.01-tf2-py3"

docker run --rm -it --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "${IMAGE}"
