# Copy to server_env.sh on a Linux GPU container and source it before any run.
# Measured on a Vast.ai L40S container (30.6-CPU quota, 255 visible host cores,
# 7,680-thread pids limit, no NVIDIA EGL library mounted).
. "$(dirname "${BASH_SOURCE[0]}")/../.venv.cuda/bin/activate"

# Rendering: Mesa EGL (software llvmpipe) with one render thread per process.
# NVIDIA EGL is faster per frame, but each context reserves ~650 MiB of GPU memory,
# so several runs with 16 environment workers each exhaust a 48 GB GPU. Keep one
# renderer for a whole study: Mesa and NVIDIA frames differ slightly (edge pixels).
export MUJOCO_GL=egl LP_NUM_THREADS=1

# JAX on GPU without preallocation, so several runs share one GPU; full float32
# matmuls so online and scanned recurrences agree as in the unit tests (no TF32).
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

# BLAS/OpenMP size thread pools to the visible host cores; one thread each keeps
# many environment workers under the container's thread limit.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
