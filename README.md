# DiscoGeo

DiscoGeo tests whether a released discovered RL rule makes geometric information
more accessible in an RGB representation, then tests transfer to a pretrained
3D generative prior. It includes actual Disco103 training and a TripoSplat
conditioning adapter, not a reimplementation of the discovered rule.

The default interaction task is active **same/different object recognition**.
The agent sees a current RGB view and a reference RGB view, can rotate left or
right, and eventually answers SAME or DIFFERENT. All identities use the same
material. Reward is simply:

```text
correct final answer: +1
incorrect answer or timeout: 0
each camera movement: -0.02
```

No depth, pose, normals, coordinates, correspondence, reconstruction loss, or
generative-teacher gradient enters RL representation training. The environment
knows identity to score the answer and view adjacency to move the camera. Those
indices are never observations. Recognition reward is task supervision.

## Losses and scientific scope

The reward is simple; **Disco103 is not a one-term policy-gradient algorithm**.
Its official KL prediction/policy losses, action-value loss, and next-policy
prediction remain intact. Removing them would change the released algorithm.
There are no extra geometry losses. `no_yz_loss` and `no_aux_policy` are explicit
perturbations of the frozen rule, not newly meta-learned algorithms.

The downstream adapter uses one MSE on TripoSplat's native image-conditioning
features. Both the RL encoder and pretrained generator stay frozen. This is a
conditioning-distillation research extension; it is **not full diffusion-model
fine-tuning**. Generated geometry can come from TripoSplat's pretrained prior.
The generation controls compare untrained/actor-critic/Disco features, native
conditioning, zero conditioning, and wrong-scene conditioning. Better generations
alone do not demonstrate emergent geometry.

The analytic object benchmark is intended for fast controlled experiments. It is
not Kubric, does not guarantee that every episode requires multiple views, and
has fixed lighting. The feedforward actor has no observation-history memory;
its action-model LSTM is not a temporal memory. A one-view shortcut and chance
recognition policy are meaningful failure modes. Pose probes are diagnostic;
held-out depth probing is stronger evidence, still limited by this synthetic domain.

## Install

Use Python 3.11 or 3.12; Python 3.12 was used for local CPU verification.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the profile needed for each workload from the repository root:

| File | Purpose |
| --- | --- |
| [`requirements.txt`](requirements.txt) | Complete Python workflow: RL, generation, plots, tests/lint |
| [`requirements-vast.txt`](requirements-vast.txt) | Python 3.12 Linux GPU profile with CUDA 12 JAX and pinned PyTorch/torchvision wheels |
| [`requirements-system.txt`](requirements-system.txt) | OS tools and runtime conditions; not a pip input |
| [`requirements-cuda.txt`](requirements-cuda.txt) | Linux NVIDIA RL training, including the matching CUDA JAX build |
| [`requirements-generative.txt`](requirements-generative.txt) | TripoSplat transfer and generation; usable in a separate environment |
| [`requirements-dev.txt`](requirements-dev.txt) | All CPU integration tests and lint, including RL and generation dependencies |
| [`requirements-gqn.txt`](requirements-gqn.txt) | Optional TensorFlow converter; preferably a separate environment |

The files select the extras declared in `pyproject.toml`, which remains the single
source for package version constraints. They use an editable install because the
pinned upstream sources and released Disco103 checkpoint are part of this checkout.

For NVIDIA training on Linux, install the matching pinned CUDA JAX build:

```bash
python -m pip install -r requirements-cuda.txt
python -c 'import jax; print(jax.devices())'
```

The repository includes the audited upstream DiscoRL source and the small
released Disco103 checkpoint. No remote rule weights need to be fetched during
training. Third-party licenses, commits and checkpoint checksum are in
[`third_party/PROVENANCE.json`](third_party/PROVENANCE.json).

## First complete run

For the entire GPU experiment, including all ablations and final 3D comparisons:

```bash
bash scripts/run_full_experiment.sh
# After interruption, repeat the same settings:
bash scripts/run_full_experiment.sh --resume
# Preview without allocating GPU work or downloading weights:
bash scripts/run_full_experiment.sh --dry-run
```

The wrapper uses Python from an active environment or `.venv.cuda`; optional
`--setup` runs the Vast installer first. Defaults are 512 procedural objects,
3 RL seeds, 500k transitions per trained variant, six encoder variants (including
untrained), snapshots every 1,000 learner updates, and four final test objects.
It prepares/reuses data and native conditioning, trains/probes every ablation,
processes validation snapshots, fits matched adapters to each **best** encoder,
then generates native/learned/zero/shuffled controls on identical test inputs.
The final stage performs 220 generations at default settings; periodic monitoring
adds its own jobs. Use `--snapshot-every 0` for final inference only.

All artifacts live under `runs/full-experiment/`: `study/report/` for RL/probes,
`transfer/report/` for final per-image/per-view/per-seed CSV/JSON, comparison PNG/PDF,
and qualitative grids; `logs/pipeline.log` for combined process output.
Views are averaged within image, images within RL seed, then variability is
reported across RL seeds. Native RGB inference is shared once per test object.
The current 3D comparison is explicitly **UNREGISTERED orbit diagnostics** using
the approximate CPU renderer. A repeated invocation with `--resume` requires
identical settings/features and reuses completed adapters/generations.

```bash
python -m disco_geo prepare --output datasets/smoke --scenes 20 --views 6 --size 16
python -m disco_geo train --config configs/smoke.json
python -m disco_geo extract --checkpoint runs/smoke/disco/best.pkl \
  --dataset datasets/smoke --output runs/smoke/disco/features.npy
python -m disco_geo probe --dataset datasets/smoke \
  --features runs/smoke/disco/features.npy --output runs/smoke/disco/probes.json \
  --label-budget 24
```

This smoke run validates plumbing; its few updates and tiny test set are not a
research result. Outputs include JSONL learning curves, best and latest
checkpoints, frozen feature arrays, provenance, and probe metrics. Existing run
directories are refused to prevent accidental overwrite.

For a real pilot:

```bash
python -m disco_geo prepare --output datasets/objects --scenes 512 --parts 8
python scripts/run_study.py --config configs/objects.json --steps 500000 \
  --seeds 0 1 2 --ablations
```

The study runs an untrained encoder, the official V-trace actor-critic,
actor-critic with the same next-policy auxiliary prediction, and Disco103. With
`--ablations`, it also runs the two Disco loss perturbations. Each run uses a
separate process so GPU memory is released. The shared CNN initialization is
matched per seed. This is fresh-trajectory training, one learner update per
rollout; the methods may still encounter different observations through their
policies. Equal trajectories and wider hyperparameter searches are subsequent
causal controls, not claimed as completed here.
The runner saves a random-policy control and `summary.json` with per-seed metrics,
means and standard deviations. Variation across three seeds is descriptive;
it should not be treated as proof of a statistically reliable effect.

The scene/shape split is 50% RL training, 10% RL validation, 20% probe training,
10% probe validation, and 10% final test. Generated shape identities are unique
under translation and proper cube rotations; every view stays with its identity.
The checkpoint is selected using RL-validation return only. Probes use frozen
single-image features before reference fusion, train-only standardization,
validation-selected ridge regularization, a fixed 64-unit relative-yaw MLP,
and scene-disjoint final tests. Depth is a linear map to a downsampled depth grid;
valid foreground masks are evaluation-only. There is no scale alignment.

The recognition random-answer accuracy is 50%; always answering SAME is not
evidence of useful learning. Report accuracy together with moves and reward.
For stable conclusions use several training seeds and larger held-out sets.

## GQN

Download a few official `rooms_ring_camera` TFRecord shards, not the whole dataset.
Run conversion in an environment with the optional `gqn` dependencies (TensorFlow
is not used by the learner):

```bash
python -m pip install -r requirements-gqn.txt
python -m disco_geo convert-gqn /path/to/0001-of-2160.tfrecord \
  --output datasets/gqn_ring --max-scenes 10000
python -m disco_geo train --config configs/gqn.json
```

The navigation task has LEFT, RIGHT and STOP, plus current/goal RGB. It uses
only terminal success and movement cost. Camera yaw constructs the private
transition graph; sorted-angle hops may have unequal angular size. The standard
GQN records have RGB and poses, so depth metrics are unavailable. GQN room views
are not suitable inputs for an object-centric TripoSplat evaluation.

## Go directly to TripoSplat transfer

The extension is implemented against the official TripoSplat **flow-matching**
pipeline, which samples 3D latents and decodes Gaussian splats. It replaces native
image conditioning with tensors predicted from frozen RL features. The public
inference API does not accept arbitrary RL features directly; the adapter is our
research extension. There is no claim that this particular method is the best
current baseline on the DiscoGeo benchmark.

Use a separate PyTorch environment/process to keep JAX from reserving GPU memory:

```bash
python -m pip install -r requirements-generative.txt
python -m disco_geo download-triposplat --output checkpoints/triposplat
python -m disco_geo cache-conditioning --dataset datasets/objects \
  --weights checkpoints/triposplat --output datasets/tripo_cache --max-items 128
```

Caching uses RGB only and pretrained DINOv3/FLUX.2 conditioning. It never reads
the hidden geometric annotations. Cache training, validation and testing use
different identities. Features must first be extracted for every encoder under
comparison; this checks checkpoint/dataset provenance.

```bash
python -m disco_geo train-adapter \
  --features runs/study/disco-none/seed0/features.npy \
  --cache datasets/tripo_cache --output runs/adapters/disco_seed0
```

Repeat adapter training for the actor-critic and untrained features with the
same cache, seed and budget. The small adapter can train on CPU. Native targets
have thousands of tokens; random token subsets make each fitting step cheaper.

To run the complete paired transfer experiment automatically, use actual test
items from the cache rather than manually choosing image/scene numbers:

```bash
python scripts/run_transfer.py --cache datasets/tripo_cache \
  --weights checkpoints/triposplat --output runs/transfer_seed0 \
  --features disco=runs/study/disco-none/seed0/features.npy \
    ac=runs/study/actor_critic-none/seed0/features.npy \
    untrained=runs/study/untrained-none/seed0/features.npy \
  --test-items 4
```

This fits each adapter with the same seed and budget, generates one native-image
baseline per test input, then learned/zero/shuffled controls for each encoder.
Every generation runs in its own process. `plan.json` records the selected input
identities and settings. Optional `--diagnostic-dataset datasets/objects` also
computes unregistered orbit metrics; registration is still needed for geometric
interpretation. The commands below are for inspecting individual cases.

Choose a **test** item from `datasets/tripo_cache/manifest.json`. Its `scene`,
`view`, and original image path identify the common source input. For example,
replace the example scene/view below with that actual item:

```bash
python -m disco_geo generate --weights checkpoints/triposplat \
  --mode native --image datasets/tripo_cache/000110.png \
  --output runs/generation/native --seed 42

python -m disco_geo generate --weights checkpoints/triposplat \
  --mode learned --features runs/study/disco-none/seed0/features.npy \
  --adapter runs/adapters/disco_seed0/best.pt --scene 480 --view 0 \
  --output runs/generation/disco --seed 42
```

Do not use the example numbers blindly: cache size can be less than requested
when there are fewer unique identities. Use exactly the manifest's item values.
Repeat with `--mode zero` and `--mode shuffled`, and with each baseline adapter.
All use the same flow noise seed and sampling settings. Native-image encoding
has a separate random generator so it does not consume the flow-noise stream.
The octree decoder's global RNG is reset separately for paired point sampling.
Learned conditioning does not use native image features at generation time.

Default outputs are `object.ply`, activated `gaussians.npz`, and
`generation.json`, plus the raw sampled camera tensor for later inspection.
Add `--exports ply npz splat` when the additional viewer format is needed.
Metadata records process timing, sampling time and peak PyTorch CUDA memory.
Generation starts with 32,768 Gaussians; higher counts cost
more to decode and render. Learned-mode flow and decoder are loaded sequentially.
Pretrained checkpoints are not bundled or automatically downloaded.

Held-out orbit diagnostics are available for the analytic object dataset:

```bash
python -m disco_geo evaluate-generation --generation runs/generation/disco \
  --dataset datasets/objects --scene 480 --source-view 0 \
  --output runs/generation/disco/evaluation --allow-unregistered
```

TripoSplat's internal coordinate frame may differ from the benchmark's frame.
Unregistered metrics are explicitly diagnostics. For an interpretable comparison,
supply `--alignment verified_transform.json`, a 4x4 similarity matrix obtained
without optimizing against held-out test views. This evaluator uses a small CPU
anisotropic Gaussian renderer and reports full-image PSNR, foreground PSNR and
silhouette IoU. It does not claim numerical parity with the official rasterizer.
The source view is excluded. The Gaussian NPZ uses the internal frame; exported
PLY/SPLAT files apply the upstream viewer-coordinate transform.

## Server and verification

For the combined pilot: L40S 48 GB, 32–64 GB host RAM, about 8 vCPUs and
100 GB writable SSD are reasonable starting allocations. With retained essentials
and small datasets, 50–80 GB may suffice. Full-model fine-tuning is outside this
implementation. Actual pretrained inference peak VRAM and speed must be measured
on the target GPU; they have not been verified locally.
The five required pretrained files total approximately 3.78 GB before caches;
128 float16 conditioning targets need roughly 1.5 GB uncompressed.
Only best/latest full RL checkpoints, compact periodic CNN snapshots, metrics,
features, adapter best weights, selected 3D outputs and interpretation plots are retained.

The [Korean Vast/L40S guide](docs/VAST_L40S_KO.md) covers upload, installation,
tmux, Inactive/Stop semantics, resume, storage and conditional runtime estimation.
`bash scripts/setup_vast.sh` prepares a Python 3.12 GPU environment; Conda is optional.
Use `benchmark-rl` on the actual host before budgeting a full study.

```bash
python scripts/run_study.py --config configs/objects.json --steps 500000 \
  --seeds 0 1 2 --ablations --snapshot-every 1000 --snapshot-probes
python -m disco_geo report --study runs/study
```

Study reporting writes numeric CSV/JSON plus PNG/PDF comparisons, individual
seed learning curves, validation snapshot probes and depth prediction panels.
For periodic 3D, replace `--snapshot-probes` with `--snapshot-cache datasets/tripo_cache`
and `--snapshot-weights checkpoints/triposplat`. Each run first trains RL, then
processes its saved iterations in separate processes on fixed **probe_val** objects.
Each snapshot gets a freshly fitted adapter with matched settings; the RL encoder
and losses remain unchanged. These are postprocessed iteration outputs, not concurrent
3D inference during RL. `scripts/run_snapshots.py --plan-only` previews selected jobs.
Resume with `--resume`; `latest.pkl` restores learner/optimizer, actor/environment state
and RNG. Final tests remain separate from iteration monitors.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check disco_geo scripts tests
```

Tests verify scene separation, no annotation-file reads by the training loader,
camera adjacency, sparse rewards/terminal resets, exact native head shapes,
finite compiled Disco103 updates with changing CNN weights and fixed rule weights,
matched CNN initialization, a real official TripoSplat transformer receiving
adapter conditioning, encoder-gradient isolation, and probe/renderer conventions.
The reduced official model also runs flow sampling, octree/Gaussian decoding,
PLY/SPLAT export and repeatable point sampling; it has random weights and does
not verify the pretrained model's output quality.

Local CPU tests and smoke training do not validate full pretrained generation
quality, CUDA memory use, GQN conversion, or the emergence hypothesis. A real
multi-seed GPU experiment and held-out generative evaluation are still required.

## Sources

- [DiscoRL official repository](https://github.com/google-deepmind/disco_rl)
- [DiscoRL paper](https://www.nature.com/articles/s41586-025-09761-x)
- [TripoSplat official inference](https://github.com/VAST-AI-Research/TripoSplat)
- [TripoSplat paper](https://arxiv.org/abs/2605.16355)
- [TripoSplat full training code](https://github.com/runjie-yan/TripoSplat-Training)
- [GQN datasets](https://github.com/google-deepmind/gqn-datasets)
