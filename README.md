# DiscoSpatial

**English** | [한국어](README_ko.md)

**Which spatial abilities form under a limited reward?** We apply the released Disco103 learning rule to a recurrent RGB agent and study how experience reuse and hindsight feedback affect each ability.

The main benchmark is **Memory Maze**. Only the original task's +1 reward for touching the requested target is used; maps, positions, depth and shortest paths are never given to the actor or the teacher. The TripoSplat and earlier object-recognition experiments were separated from the active study. [Recovering the previous experiments](docs/LEGACY_ARCHIVE.md)

[Research design: spatial abilities and SDPO-style feedback distillation](docs/SDPO_SPATIAL_RESEARCH_PLAN.md) · [Research protocol](docs/RESEARCH_PROTOCOL.md) · [Implementation and training conventions](docs/METHODS.md) · [Condition table](experiments/conditions.json)

## What is implemented and verified

| Item | Status |
| --- | --- |
| Real Memory Maze RGB-only wrapper, separated evaluation sidecar | Implemented; real reset/step checked locally |
| Evaluation render telemetry (segmentation visibility, depth-backprojected known cells, coarse depth) | Implemented; re-rendered RGB matches the observation pixel for pixel; 99.8% of floor and 99.7% of wall pixels land in the right cells |
| CNN + previous action/reward + GRU + sequential unroll | Implemented; online/unroll, reset and history-gradient checks; block-wise prefix reconstruction |
| Released Disco103 / AC / AC + next-policy auxiliary | Implemented; learner checked with the real frozen rule |
| Fresh, recent sequence replay, positive-advantage self-imitation | Implemented; smoke runs on the real benchmark |
| SDPO-style feedback self-teacher (outcome / outcome + success memory, shuffled control, evidence gate, teacher acting) | **Implemented, usefulness not verified**; Bayes-posterior recovery unit test, smoke runs; an offline check with a fixed explore policy found no held-out information gain (gate closed) |
| Six-family ability suite, navigation (primary 3), cross-seed report | Implemented; run on real data at smoke scale |
| Performance or emergence claims for the new study | None yet; smoke runs check wiring and are not research results ([summary](reports/development/smoke_summary.json)) |

## Research questions and evaluation

- **H1:** With the same recurrent backbone and limited reward, does Disco preserve and use spatial information better than AC+aux?
- **H2:** Beyond simply reusing past rewards, does feedback-conditioned distillation add value?
- **H3:** Is the teacher's improvement signal due to useful hindsight feedback, or to new supervision or extra compute?

Abilities are grouped into six families: self-localization and motion estimation, memory of objects not currently visible, place recognition, maps and connectivity, local geometry and affordances, and actual navigation. The primary endpoints are **seen-but-hidden target location, the causally known wall map, and matched revisit efficiency**. OOD is not an additional ability; it is a generalization axis of these abilities.

We separate three things: information that a frozen probe can read, transfer to a new controller, and use by the frozen actor itself. Splits are by layout and episode, statistics by seed and layout, and frames are never counted as independent samples. The planned five seeds are for the pilot; the confirmatory sample size will be fixed separately after a power plan. A generic readout without full visibility telemetry is not called a primary ability evaluation.

## Model and training

```text
RGB → CNN(SiLU) → signed projection + LayerNorm → image feature
                       + previous action / previous reward
                       → GRU → belief state → heads of each official algorithm
```

The Disco103 update-rule weights are frozen; the visual encoder, GRU and policy/prediction/value heads are trained. The action-model LSTM is separate from the actor GRU. `is_first` is stored, and a genuine sequential unroll runs from the rollout's starting hidden state. LAST is used only for bootstrapping, and the original continuation discount is preserved.

Recent replay stores raw episode sequences. The current encoder and recurrent state are recomputed from the episode start, and the original behavior logits are kept. Fresh and replay gradients are mixed in one update, and the meta-rule state and EMAs are updated once from the fresh stream only. Replay age and excessive log-ratios are limited, and this convention itself is recorded as an extension of Disco.

The SDPO-style distillation is `L = L_Disco,fresh [+ α·L_replay] + λ_SIL·L_SIL + λ_T·L_teacher + λ_SD·gate·KL(student ‖ stopgrad(teacher_EMA))`. The teacher is a residual head `base + Δ(belief, feedback)` on the student's belief (stop-gradient). It sees only the reward outcome of the agent's own attempt and the earlier success frame for the same target color in the same episode. The trained Δ estimates log P(outcome | history, action), so a success query is a soft policy improvement whose expectation equals self-imitation (the prediction for H2). Distillation opens only when held-out information beats the shuffled control. [Design and rationale](docs/SDPO_SPATIAL_RESEARCH_PLAN.md) · [Detailed conventions](docs/METHODS.md)

## Installation

Use Python **3.12**. The environment stack below was used for local macOS CPU wiring checks. Linux CUDA and EGL are checked separately on the target host.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m disco_spatial --help
python -m disco_spatial protocol --config configs/development.json
```

Memory Maze 1.0.3 requires Gym at import even on the dm_env path, so Gym is pinned explicitly; execution uses the dm_env interface. Dependency versions are pinned in `pyproject.toml`. Memory Maze's border drawing covers the whole image below 64 pixels, so the real environment enforces at least 64 pixels.

Local rendering check on macOS:

```bash
MUJOCO_GL=glfw python -m disco_spatial doctor --environment \
  --output runs/environment/doctor.json
```

Linux NVIDIA server:

```bash
bash scripts/setup_gpu.sh
cp scripts/server_env.example.sh server_env.sh && . ./server_env.sh
python -m disco_spatial doctor --environment --output runs/environment/doctor.json
```

Findings on a containerized GPU server (Vast.ai L40S, 2026-10-28):

- **Environment speed.** Environment stepping sets the training speed: one environment step took 19–29 ms on the server and 4 ms on the Mac. The pilot config sets `env_processes=true`, which runs one worker process per environment. That raised one run from 23 to about 200 steps per second; 5M steps take about 7 hours.
- **Thread limit.** XLA and OpenBLAS create as many threads as there are visible host cores (255). `server_env.sh` fixes BLAS threads to 1 so the container's thread limit (7,680) is not exceeded, and `scripts/run_stage.sh` gives each run its own block of cores with `taskset`.
- **Renderer.** One NVIDIA EGL context takes about 650 MiB of GPU memory, so 4 runs × 16 workers fill a 48 GB GPU. All server runs therefore use Mesa EGL. The two renderers differ slightly at edge pixels, so they are not mixed within one study.
- **Running a stage.** `bash scripts/run_stage.sh <config directory> 4 8` runs the configs four at a time, lowest seed first, with 8 cores each.

Monitoring and evaluating a stage:

```bash
python3 scripts/stage_status.py runs/pilot
python scripts/evaluate_stage.py runs/pilot --parallel 4 --cores 8
python scripts/expand_conditions.py --stage P2_pilot_matrix --skip-finished --output runs/pilot2/configs
```

`stage_status.py` prints each run's steps, speed, ETA, validation return, teacher gate and failures. `evaluate_stage.py` collects the common probe histories once and then evaluates every finished run: ability suite, navigation with memory interventions, teacher acting, and the untrained and random baselines. It writes per-method reports and packs everything into `runs/<stage>_results.tar.gz`. It skips finished work, so it can be rerun after an interruption. `--skip-finished` lets a later stage reuse runs that an earlier stage already finished.

## Development runs

`integration.json` is a wiring check of **only 8 transitions**. It has no validation, so it creates no best checkpoint.

```bash
python -m disco_spatial train --config configs/integration.json \
  --output runs/integration/disco-fresh
python -m disco_spatial train --config configs/integration.json \
  --recipe recent_replay --output runs/integration/disco-replay
```

Longer development runs name their config explicitly. Existing directories are never overwritten.

```bash
python -m disco_spatial train --config configs/development.json
python -m disco_spatial train --config configs/replay_development.json
```

Self-imitation and the feedback teacher are enabled through config fields or CLI overrides. `scripts/expand_conditions.py` expands the full condition table into per-seed configs.

```bash
python -m disco_spatial train --config configs/smoke.json --self-imitation-weight 0.5 \
  --output runs/smoke/disco-sil
python -m disco_spatial train --config configs/smoke.json --teacher outcome_memory \
  --teacher-query counterfactual_success --distillation-weight 0.5 --output runs/smoke/disco-fd-memory
python scripts/expand_conditions.py --stage P1_learning_gate --output runs/pilot/configs
```

Whole real episodes, the memory-reset intervention and teacher acting:

```bash
python -m disco_spatial evaluate --checkpoint runs/development/disco-seed0/best.pkl \
  --seeds 1000000 1000001 --output runs/evaluation/intact.json
python -m disco_spatial evaluate --checkpoint runs/development/disco-seed0/best.pkl \
  --seeds 1000000 1000001 --reset-memory --output runs/evaluation/reset.json
python -m disco_spatial navigate --checkpoint runs/development/disco-seed0/best.pkl \
  --seeds 2000000 2000001 --interventions none reset_every_step truncate_16 \
  --output runs/navigation/disco-seed0.json
```

## Common-history ability suite

Each method's representation is compared on the same observation histories. Collection uses a fixed exploratory reference policy (`--policy explore`) and render telemetry. Validation is collected separately from seeds in the 4,000,000 range and the final test from the 5,000,000 range.

```bash
python -m disco_spatial collect --partition probe_train --seed 3000000 \
  --episodes 4 --max-steps 1000 --output datasets/probe_train
python -m disco_spatial extract --checkpoint runs/development/disco-seed0/best.pkl \
  --collection datasets/probe_train --output runs/features/probe_train
```

After extracting all three partitions with the same checkpoint, run the six-family suite. Also extract each run's `init.pkl` as an untrained control with the same initialization.

```bash
python -m disco_spatial abilities --train runs/features/probe_train \
  --validation runs/features/probe_validation --test runs/features/probe_final_test \
  --output runs/abilities/disco-seed0.json
python -m disco_spatial report --reference acaux \
  --runs acaux:0:runs/abilities/acaux-seed0.json:runs/navigation/acaux-seed0.json \
         disco:0:runs/abilities/disco-seed0.json:runs/navigation/disco-seed0.json \
  --output runs/reports/pilot.json
```

The layout fingerprint treats grid rotations and reflections as identical and refuses to fit if partitions actually overlap. This checks geometry separation between probe partitions. The RL-train geometry inventory is not connected yet, so we **do not claim complete separation between RL-train and all probe layouts**.

## Reproducibility and remaining research work

Each run saves its config, protocol hash, source hashes, environment versions, learner log, typed raw trajectories and checkpoints. `latest.pkl` is a policy/optimizer snapshot. **Exact resumption of simulator and replay state is not supported yet**; until a recoverable full-state resume exists, start runs in a new output directory.

Confirmatory mode is disabled. `P1_learning_gate` (above-random learning, a drop under memory reset, teacher evidence) finished on 2026-10-30 and **did not pass G2**: every run's policy collapsed to uniform, and the teacher's Δ stayed at zero. `P1d_diagnostics` (learner positive controls, credit diagnostics, AC+aux entropy and Disco103 learning-rate ablations) comes next. Local checks already show that Disco103 at P1's learning rate (3e-4) fails even a one-answer bandit, in the unmodified upstream harness too, but learns it at 3e-3–1e-2; `P2_pilot_matrix` with 5 seeds, a power plan and the design lock wait for it. A guided study that gives the teacher GT maps or poses is kept separate from the main study.

```bash
python -m pytest -q
ruff check disco_spatial tests
bash -n scripts/setup_gpu.sh
```

## Sources

- [Memory Maze](https://github.com/jurgisp/memory-maze), [paper](https://arxiv.org/abs/2210.13383)
- [DiscoRL](https://github.com/google-deepmind/disco_rl), [pinned source and checkpoint provenance](third_party/PROVENANCE.json)
- [SDPO](https://arxiv.org/abs/2601.20802), [official code](https://github.com/lasgroup/SDPO)

Last development update: 2026-10-28.
