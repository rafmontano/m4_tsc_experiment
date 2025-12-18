#!/usr/bin/env bash
set -euo pipefail

#IMAGE="nvcr.io/nvidia/tensorflow:25.01-tf2-py3"
IMAGE="m4_tf_inception:25.01"

SCRIPT="${1:-src/python/gpu_test/10_inceptiontime_smoke_docker.py}"

docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "${IMAGE}" \
  python "${SCRIPT}"
