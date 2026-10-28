#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Linux ]]; then
  echo "This script is for a Linux NVIDIA host. Use the README macOS setup locally." >&2
  exit 1
fi
nvidia-smi
python3.12 -m venv .venv.cuda
source .venv.cuda/bin/activate
python -m pip install -r requirements-cuda.txt
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
python -c 'import jax; print(jax.devices()); assert jax.default_backend() == "gpu"'
python -m disco_spatial doctor --environment --output runs/environment/doctor.json
