#!/usr/bin/env bash
# Run from this checkout on the rented Linux GPU instance.
set -euo pipefail
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: bash scripts/setup_vast.sh [--skip-system]"
  echo "Requires Linux x86_64, Python 3.12, a CUDA-ready NVIDIA host. Conda is optional."
  exit 0
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This installer targets Linux x86_64 Vast instances." >&2
  exit 1
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
nvidia-smi -L
task_python="${DISCOGEO_PYTHON:-python3.12}"
if ! command -v "$task_python" >/dev/null; then
  task_python=python
fi
"$task_python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Choose a Python 3.12 image, or use environment.yml"'
if [[ "${1:-}" != "--skip-system" ]]; then
  task_missing=()
  for task_tool in git tmux rsync; do
    if ! command -v "$task_tool" >/dev/null; then task_missing+=("$task_tool"); fi
  done
  if ((${#task_missing[@]})); then
    task_privilege=()
    if [[ "$(id -u)" != 0 ]]; then task_privilege=(sudo); fi
    "${task_privilege[@]}" apt-get update
    "${task_privilege[@]}" apt-get install -y "${task_missing[@]}" ca-certificates
  fi
fi
if ! "$task_python" -m venv .venv.cuda; then
  echo "Install the matching python3.12-venv apt package, or use optional Conda." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv.cuda/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements-vast.txt
nvidia-smi
# Separate processes let each framework release its GPU allocations.
python - <<'PY'
import jax
print("JAX", jax.__version__, jax.devices())
assert jax.default_backend() == "gpu", "JAX fell back to CPU; inspect the host driver"
jax.block_until_ready(jax.numpy.ones((8, 8)) @ jax.numpy.ones((8, 8)))
PY
python - <<'PY'
import torch
print("PyTorch", torch.__version__, "wheel CUDA", torch.version.cuda)
assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
print(torch.cuda.get_device_name(), "BF16", torch.cuda.is_bf16_supported())
torch.ones((8, 8), device="cuda") @ torch.ones((8, 8), device="cuda")
torch.cuda.synchronize()
PY
mkdir -p runs/environment
python -m pip freeze --all > runs/environment/pip-freeze.txt
nvidia-smi > runs/environment/nvidia-smi.txt
python -c 'import sys, platform; print(sys.version); print(platform.platform())' > runs/environment/runtime.txt
echo "Ready. Activate with: source .venv.cuda/bin/activate"
