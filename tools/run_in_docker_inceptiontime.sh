#!/usr/bin/env bash
set -euo pipefail

IMAGE="m4_tf_inception:25.01"
SCRIPT="${1:-src/python/run_experiment_inceptiontime_all.py}"

docker run --rm --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e TF_CPP_MIN_LOG_LEVEL=2 \
  -e PYTHONPATH=/workspace \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "${IMAGE}" \
  python -u "${SCRIPT}"
