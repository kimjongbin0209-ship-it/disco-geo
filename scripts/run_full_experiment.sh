#!/usr/bin/env bash
# One entry point: data -> ablations -> frozen adapters -> final paired inference -> reports.
set -euo pipefail
task_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo"

usage() {
  cat <<'HELP'
Usage: bash scripts/run_full_experiment.sh [options]
  --setup                    Install the Vast GPU environment first (optional)
  --python PATH              Python executable (default: active env / .venv.cuda / python3)
  --config FILE              Training architecture/config (default: configs/objects.json)
  --output DIR               All study/transfer/reports/logs (default: runs/full-experiment)
  --dataset DIR              Override config dataset; reuse if already prepared
  --cache DIR                Native conditioning cache (default: datasets/tripo_cache)
  --weights DIR              Pretrained weights (default: checkpoints/triposplat)
  --steps N                  RL transitions per trained run (default: 500000)
  --seeds N ...              RL seeds (default: 0 1 2)
  --test-items N             Same final test objects for every model (default: 4)
  --adapter-epochs N         Matched adapter budget (default: 30)
  --snapshot-every N         Save/process RL iterations; 0 = final inference only (default: 1000)
  --snapshot-items N         Fixed validation objects per iteration (default: 2)
  --cache-items N            Native cache budget (default: 128)
  --label-budget N           Probe label budget (default: 5000)
  --scenes N --views N --size N --parts N --data-seed N
                             New dataset only (defaults: 512 / 12 / 64 / 8 / 0)
  --resume                   Reuse completed stages; require identical pipeline settings
  --dry-run                  Print the planned commands; no install/download/training
  --help                     Show this help
Final inference compares all 6 encoder variants per RL seed with native/learned/zero/shuffled.
Run inside tmux on a Running GPU instance. Output comparisons are UNREGISTERED diagnostics.
HELP
}

task_config=configs/objects.json
task_output=runs/full-experiment
task_dataset=""
task_cache=datasets/tripo_cache
task_weights=checkpoints/triposplat
task_steps=500000
task_seeds=(0 1 2)
task_test_items=4
task_epochs=30
task_snapshot_every=1000
task_snapshot_items=2
task_cache_items=128
task_label_budget=5000
task_scenes=512
task_views=12
task_size=64
task_parts=8
task_data_seed=0
task_resume=0
task_dry=0
task_setup=0
task_python="${DISCOGEO_PYTHON:-}"
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --setup) task_setup=1; shift ;;
    --resume) task_resume=1; shift ;;
    --dry-run) task_dry=1; shift ;;
    --seeds)
      shift; task_seeds=()
      while (($#)) && [[ "$1" != --* ]]; do task_seeds+=("$1"); shift; done
      ((${#task_seeds[@]})) || { echo "--seeds requires at least one integer" >&2; exit 2; }
      ;;
    --python|--config|--output|--dataset|--cache|--weights|--steps|--test-items|--adapter-epochs|--snapshot-every|--snapshot-items|--cache-items|--label-budget|--scenes|--views|--size|--parts|--data-seed)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --python) task_python="$2" ;; --config) task_config="$2" ;;
        --output) task_output="$2" ;; --dataset) task_dataset="$2" ;;
        --cache) task_cache="$2" ;; --weights) task_weights="$2" ;;
        --steps) task_steps="$2" ;; --test-items) task_test_items="$2" ;;
        --adapter-epochs) task_epochs="$2" ;; --snapshot-every) task_snapshot_every="$2" ;;
        --snapshot-items) task_snapshot_items="$2" ;; --cache-items) task_cache_items="$2" ;;
        --label-budget) task_label_budget="$2" ;; --scenes) task_scenes="$2" ;;
        --views) task_views="$2" ;; --size) task_size="$2" ;; --parts) task_parts="$2" ;;
        --data-seed) task_data_seed="$2" ;;
      esac
      shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
for task_number in "$task_steps" "$task_test_items" "$task_epochs" "$task_snapshot_every" "$task_snapshot_items" "$task_cache_items" "$task_label_budget" "$task_scenes" "$task_views" "$task_size" "$task_parts" "$task_data_seed" "${task_seeds[@]}"; do
  [[ "$task_number" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Expected a nonnegative integer: $task_number" >&2; exit 2; }
done
for task_number in "$task_steps" "$task_test_items" "$task_epochs" "$task_snapshot_items" "$task_label_budget"; do
  ((task_number > 0)) || { echo "Budgets/counts must be positive" >&2; exit 2; }
done
((task_scenes >= 10 && task_views >= 3 && task_size >= 8 && task_parts >= 5 && task_cache_items >= 3)) || { echo "Invalid dataset/cache size" >&2; exit 2; }
task_seen=" "
for task_seed in "${task_seeds[@]}"; do
  [[ "$task_seen" != *" $task_seed "* ]] || { echo "Duplicate seed: $task_seed" >&2; exit 2; }
  task_seen+="$task_seed "
done
if [[ -z "$task_python" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then task_python="$VIRTUAL_ENV/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" ]]; then task_python="$CONDA_PREFIX/bin/python"
  elif [[ -x "$task_repo/.venv.cuda/bin/python" ]]; then task_python="$task_repo/.venv.cuda/bin/python"
  elif command -v python3.12 >/dev/null; then task_python=python3.12
  else task_python=python3; fi
fi
if [[ -z "$task_dataset" ]]; then
  task_dataset="$("$task_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset"])' "$task_config")"
fi
[[ -n "$task_output" && -n "$task_dataset" && -n "$task_cache" && -n "$task_weights" ]] || { echo "Paths must not be empty" >&2; exit 2; }
task_resolved="$task_output/experiment-config.json"
task_stage=""
cleanup() { if [[ -n "$task_stage" && -d "$task_stage" ]]; then rm -rf -- "$task_stage"; fi; }
trap cleanup EXIT
trap 'echo "Pipeline interrupted/failed. Re-run the same options with --resume; inspect the log." >&2' ERR
run() {
  printf '[%s] ' "$(date -u +%FT%TZ)"
  printf '%q ' "$@"
  printf '\n'
  if ((task_dry == 0)); then "$@"; fi
}

# Record semantic settings before allocating GPU work, and reject mismatched resumes.
run "$task_python" - "$task_config" "$task_output" "$task_dataset" "$task_cache" "$task_weights" \
  "$task_steps" "$task_test_items" "$task_epochs" "$task_snapshot_every" "$task_snapshot_items" \
  "$task_cache_items" "$task_label_budget" "$task_scenes" "$task_views" "$task_size" "$task_parts" \
  "$task_data_seed" "$task_resume" "${task_seeds[@]}" <<'PY'
import json, sys
from pathlib import Path
from disco_geo.io import write_json
config, output, dataset, cache, weights, *numbers = sys.argv[1:]
steps, test_items, epochs, snapshot_every, snapshot_items, cache_items, label_budget, scenes, views, size, parts, data_seed, resume, *seeds = map(int, numbers)
root = Path(output).resolve()
if root in (Path.cwd(), Path('/')):
    raise ValueError('Choose a dedicated experiment output directory')
cfg = json.loads(Path(config).read_text())
cfg.update(dataset=str(Path(dataset).resolve()), steps=steps)
settings = dict(training_config=cfg, seeds=seeds, cache=str(Path(cache).resolve()), weights=str(Path(weights).resolve()),
                test_items=test_items, adapter_epochs=epochs, snapshot_every=snapshot_every, snapshot_items=snapshot_items,
                cache_items=cache_items, label_budget=label_budget,
                prepare=dict(scenes=scenes, views=views, size=size, parts=parts, seed=data_seed))
manifest = root / 'pipeline.json'
if manifest.exists():
    if not resume or json.loads(manifest.read_text()) != settings:
        raise ValueError('Use --resume with identical settings, or choose a fresh --output')
elif root.exists() and any(root.iterdir()):
    raise FileExistsError('Output is not a recognized full-pipeline directory; choose a new one')
else:
    write_json(manifest, settings)
write_json(root / 'experiment-config.json', cfg)
PY
if ((task_dry == 0)); then
  mkdir -p "$task_output/logs"
  exec > >(tee -a "$task_output/logs/pipeline.log") 2>&1
fi
if ((task_setup)); then
  run env DISCOGEO_PYTHON="$task_python" bash scripts/setup_vast.sh
  task_python="$task_repo/.venv.cuda/bin/python"
fi
run "$task_python" -c 'import sys,jax; assert sys.version_info[:2] == (3,12), "Use Python 3.12"; assert jax.default_backend() == "gpu", "JAX must use GPU"; print(jax.devices())'
run "$task_python" -c 'import torch,matplotlib; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA/BF16 required"; print(torch.cuda.get_device_name())'

if [[ ! -f "$task_dataset/manifest.json" ]]; then
  if ((task_dry == 0)); then
    [[ ! -e "$task_dataset" ]] || { echo "Unrecognized/incomplete dataset: $task_dataset" >&2; exit 1; }
    mkdir -p "$(dirname "$task_dataset")"
    task_stage="$(mktemp -d "${task_dataset}.stage.XXXXXX")"
    run "$task_python" -m disco_geo prepare --output "$task_stage" --scenes "$task_scenes" --views "$task_views" --size "$task_size" --parts "$task_parts" --seed "$task_data_seed"
    mv "$task_stage" "$task_dataset"; task_stage=""
  else
    run "$task_python" -m disco_geo prepare --output "$task_dataset" --scenes "$task_scenes" --views "$task_views" --size "$task_size" --parts "$task_parts" --seed "$task_data_seed"
  fi
fi
if ((task_dry)) || ! "$task_python" - "$task_weights" <<'PY'
from pathlib import Path
import sys
from disco_geo.generative.external import WEIGHTS
sys.exit(0 if all((Path(sys.argv[1]) / file).is_file() for file in WEIGHTS.values()) else 1)
PY
then
  run "$task_python" -m disco_geo download-triposplat --output "$task_weights"
fi
if [[ ! -f "$task_cache/manifest.json" ]]; then
  if ((task_dry == 0)); then
    [[ ! -e "$task_cache" ]] || { echo "Unrecognized/incomplete cache: $task_cache" >&2; exit 1; }
    mkdir -p "$(dirname "$task_cache")"
    task_stage="$(mktemp -d "${task_cache}.stage.XXXXXX")"
    run "$task_python" -m disco_geo cache-conditioning --dataset "$task_dataset" --weights "$task_weights" --output "$task_stage" --max-items "$task_cache_items" --seed "$task_data_seed"
    mv "$task_stage" "$task_cache"; task_stage=""
  else
    run "$task_python" -m disco_geo cache-conditioning --dataset "$task_dataset" --weights "$task_weights" --output "$task_cache" --max-items "$task_cache_items" --seed "$task_data_seed"
  fi
fi
run "$task_python" - "$task_dataset" "$task_cache" "$task_weights" "$task_test_items" "$task_snapshot_items" "$task_snapshot_every" <<'PY'
import json, sys
from pathlib import Path
from disco_geo.data.bundle import RGBBundle
from disco_geo.generative.external import weight_paths
from disco_geo.io import sha256
dataset, cache, weights, test_items, snapshot_items, snapshot_every = sys.argv[1:]
bundle = RGBBundle(dataset)
assert bundle.manifest['source'] == 'analytic_voxel_objects', 'This pipeline evaluates the analytic object benchmark'
manifest = json.loads((Path(cache) / 'manifest.json').read_text())
assert manifest['dataset_manifest_sha256'] == sha256(Path(dataset) / 'manifest.json'), 'Cache dataset mismatch'
paths = weight_paths(weights)
assert set(manifest['teacher_weight_sha256']) == {'dinov3_path', 'flux2_vae_encoder_path', 'rmbg_path'}, 'Incomplete teacher provenance'
assert manifest['teacher_weight_sha256'] == {k: sha256(paths[k]) for k in manifest['teacher_weight_sha256']}, 'Cache teacher weights changed'
items = manifest['items']
assert len([i for i in items if i['split'] == 'test']) >= int(test_items), 'Not enough cached test inputs'
if int(snapshot_every):
    assert len([i for i in items if i['split'] == 'probe_val']) >= int(snapshot_items), 'Not enough monitor inputs'
for item in items:
    assert item['scene'] in bundle.manifest['splits'][item['split']], 'Wrong identity split'
    assert item['scene_id'] == bundle.scene_ids[item['scene']], 'Wrong scene identity'
    assert (Path(cache) / item['conditioning']).is_file() and (Path(cache) / item['image']).is_file(), 'Incomplete cache item'
PY

task_cmd=("$task_python" scripts/run_study.py --config "$task_resolved" --output "$task_output/study" --steps "$task_steps" --seeds "${task_seeds[@]}" --ablations --label-budget "$task_label_budget" --snapshot-every "$task_snapshot_every")
if ((task_snapshot_every)); then
  task_cmd+=(--snapshot-cache "$task_cache" --snapshot-weights "$task_weights" --snapshot-items "$task_snapshot_items" --snapshot-epochs "$task_epochs")
fi
if ((task_resume)); then task_cmd+=(--resume); fi
run "${task_cmd[@]}"
task_features=()
for task_seed in "${task_seeds[@]}"; do
  for task_variant in untrained-none actor_critic-none actor_critic_aux-none disco-none disco-no_yz_loss disco-no_aux_policy; do
    task_features+=("${task_variant}_seed${task_seed}=$task_output/study/$task_variant/seed$task_seed/features.npy")
  done
done
task_cmd=("$task_python" scripts/run_transfer.py --cache "$task_cache" --weights "$task_weights" --output "$task_output/transfer" --features "${task_features[@]}" --test-items "$task_test_items" --adapter-epochs "$task_epochs" --adapter-seed 0 --sampling-seed 42 --steps 20 --num-gaussians 32768 --device cuda --adapter-device cuda --diagnostic-dataset "$task_dataset")
if ((task_resume)); then task_cmd+=(--resume); fi
run "${task_cmd[@]}"
run "$task_python" -m disco_geo report --study "$task_output/study"
run "$task_python" -m disco_geo report-transfer --transfer "$task_output/transfer"
run "$task_python" - "$task_output" <<'PY'
from pathlib import Path
import sys
from disco_geo.io import write_json
root = Path(sys.argv[1])
write_json(root / 'completed.json', dict(study_report=str(root / 'study/report'),
           transfer_report=str(root / 'transfer/report'), final_checkpoint='best.pkl', comparison='UNREGISTERED orbit diagnostics'))
PY
if ((task_dry)); then
  echo "Dry-run complete; no experiment artifacts were written."
else
  echo "Done: $task_output/study/report and $task_output/transfer/report"
fi
