#!/usr/bin/env bash
# Train every config in a directory, N at a time, lowest seed first; one log per run.
# Prepare the environment first (e.g. `. ./server_env.sh` on the GPU host).
# Usage: bash scripts/run_stage.sh runs/pilot/configs [parallel=4] [cores_per_run=8] [first_core=0]
#
# Each concurrent run gets its own slot and, where taskset exists, its own block of
# cores. XLA and OpenBLAS size thread pools to the visible cores; on a container that
# sees hundreds of host cores, unpinned runs plus environment workers exhaust the
# container's thread limit (pthread_create fails).
set -u
cd "$(dirname "$0")/.."
directory=$1
export PARALLEL=${2:-4} CORES_PER_RUN=${3:-8} FIRST_CORE=${4:-0}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SLOTS_DIR=$(mktemp -d)
export SLOTS_DIR
mkdir -p runs/logs
ls "$directory"/*.json \
  | awk -F'seed' '{split($NF, a, "."); print a[1] "\t" $0}' | sort -n -s -k1,1 | cut -f2 \
  | xargs -P "$PARALLEL" -I{} bash -c '
      name=$(basename "{}" .json)
      slot=""
      for s in $(seq 0 $((PARALLEL - 1))); do
        if mkdir "$SLOTS_DIR/$s" 2>/dev/null; then slot=$s; break; fi
      done
      pin=()
      if command -v taskset > /dev/null; then
        first=$((FIRST_CORE + slot * CORES_PER_RUN))
        pin=(taskset -c "$first-$((first + CORES_PER_RUN - 1))")
      fi
      echo "start $name slot=$slot $(date +%FT%T)"
      if "${pin[@]}" python -m disco_spatial train --config "{}" > "runs/logs/$name.log" 2>&1; then
        echo "done $name $(date +%FT%T)"
      else
        echo "FAILED $name $(date +%FT%T)"
      fi
      rmdir "$SLOTS_DIR/$slot"'
rm -rf "$SLOTS_DIR"
echo "stage finished $(date +%FT%T)"
