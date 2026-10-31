# DiscoSpatial

**English** | [한국어](README_ko.md)

**Can DeepMind's released meta-learned RL rule (Disco103) train a recurrent pixel agent on Memory Maze to form spatial abilities, and does SDPO-style hindsight feedback help?** The project stopped at its first learning gate: neither learner beat a uniform-random policy. A diagnostic stage then traced the failure to two separate causes. This page reports the work up to that stage (`P1d`). Everything here is pilot or diagnostic evidence, not a study result.

> **Status (2026-10-31):** paused after P1d for budget. 57 GPU runs in all (12 in P1, 45 in P1d) on one Vast.ai L40S. Numbers: [`reports/results/summary.json`](reports/results/summary.json) · stage reports: [P1](reports/results/p1/h1_disco_vs_acaux.md), [P1d](reports/results/p1d/diagnostics.md) · figures: [`scripts/plot_results.py`](scripts/plot_results.py)

## Summary

| Question | What the data says |
| --- | --- |
| Did either learner beat random play in Memory Maze (gate G2)? | **No.** In P1 (12 runs × 5M steps) every policy ended uniform (entropy 1.78–1.79; ln 6 = 1.79) and earned the uniform rate of about 0.2 rewards per 1000 steps. |
| Is the training pipeline broken? | **Probably not.** AC+aux solves a one-answer bandit in 3/3 seeds and a pixel cue task in 1/3. |
| Does Disco103 turn reward into policy change here? | **No.** On Memory Maze its credit gap is 0 at learning rates 3e-4, 3e-3 and 1e-2, and its policy returns to uniform. It solves the bandit only at 1e-2 (2/3 seeds) and solves no task that needs perception. |
| Why did AC+aux stay uniform? | **Its entropy bonus.** Its updates do favour actions that precede a reward (positive credit gap in all 9 maze runs), but the entropy bonus of 0.01 cancels that. At 1e-3 the reward rate is 1.56× uniform (95% CI 1.35–1.81); at 0 the policy collapses onto one action. |
| Any sign of navigation or memory? | **No.** The best learner (0.32 per 1000 steps) is level with a scripted policy that only favours `forward_left` (0.41, from 13 rewards). No run passed the 8-step memory control. Humans average 26.4 rewards per 9×9 episode. |

The bottleneck common to both learners is the budget: 2,000–5,000 updates of about 1,000 steps each, while a uniform policy finds a reward once every 5,000 steps. The self-teacher (H2, H3) was never really tested, because it needs successes that never came.

## 1. Question and setup

**Task.** Memory Maze 9×9 ([Pasukonis et al., 2022](https://arxiv.org/abs/2210.13383)) is a 3D maze seen through a 64×64 RGB camera, with 6 actions and 1000-step episodes. The agent gets +1 each time it touches the target colour it is asked for, and then a new target is requested. The maze is new every episode and fixed within it, so an agent that remembers where things are can return to targets faster. Only this reward is used. Maps, positions, depth and shortest paths never reach the actor or the teacher.

**Agent.**

```text
RGB 64×64 → CNN (32-64-64, SiLU) → projection + LayerNorm (256)
          + previous action / previous reward → GRU (256) → heads of each learner
```

**Learners** (same backbone and the same initialisation per seed):

- **Disco103**: the released, frozen meta-learned update rule from [DiscoRL](https://github.com/google-deepmind/disco_rl). The agent's network is trained; the rule that turns rollouts into learning targets is not.
- **AC+aux**: a hand-designed baseline, V-trace actor-critic with a next-policy auxiliary loss.

**SDPO-style self-teacher.** A residual head on the student's belief that also sees the outcome of the agent's own attempt and the earlier success frame for the same target. Its trained Δ estimates log P(success | history, action), so querying success gives a soft policy improvement. Distillation turns on only when the teacher's held-out information beats a shuffled-feedback control. [Design and rationale](docs/SDPO_SPATIAL_RESEARCH_PLAN.md)

**Hypotheses.**

- **H1:** Disco keeps and uses spatial information better than AC+aux.
- **H2:** feedback distillation adds value beyond reusing past rewards.
- **H3:** the teacher's gain comes from the hindsight feedback, not from extra supervision or compute.

**Gate G2** must pass before any ability comparison: above-random return, and a drop in return when the memory is reset.

## 2. P1: the learning gate failed

P1 had 12 runs: 2 learners × {`fresh`, `teacher_only`} × 3 seeds, 5M steps each (about 5,000 updates), learning rate 3e-4, AC+aux entropy cost 0.01. `teacher_only` trains the teacher but never distils from it, so its student trains like `fresh`.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/results/figures/p1_learning_gate-dark.png">
  <img src="reports/results/figures/p1_learning_gate.png" alt="P1 policy entropy and reward per 1000 steps for 12 runs over 5M steps: entropy returns to ln 6 and reward stays near the uniform-policy rate of 0.20">
</picture>

| | AC+aux (6 runs) | Disco103 (6 runs) |
| --- | --- | --- |
| Reward per 1000 steps, last quarter of training | 0.204 | 0.179 |
| Policy entropy, last quarter (uniform = 1.792) | 1.782–1.784 | 1.790–1.792 |
| Lowest policy entropy during training | 1.32–1.70 | 0.75–1.13 |

- **Early drift, then uniform.** The policies moved away from uniform early (Disco further than AC+aux) and then drifted back. One Disco run briefly favoured `forward_left` and nearly doubled its reward rate, then returned to uniform.
- **Test returns.** On 8 held-out episodes per run, returns could not be told apart from random play or the untrained network ([report](reports/results/p1/h1_disco_vs_acaux.md)).
- **Teacher (G4a).** Its held-out information gain equalled the shuffled control to within 1e-9, so Δ stayed at zero.
- **Probes.** Frozen probes after training were no better than on the untrained initialisation, and loop-closure recall fell from 0.38 to 0.31–0.33.

## 3. P1d: why did it fail?

P1d had 45 runs with 3 seeds per condition and four checks:

| Check | What it isolates | Runs |
| --- | --- | --- |
| Positive controls (`cue_recall`) | A synthetic task in which a colour cue names the rewarded action. **Bandit:** the cue is fixed, so no perception is needed. **Reactive:** the action is scored on the cue step. **Memory:** cue, 8 blank steps, then the scored step. A learner that fails the bandit has a learner problem, not a vision or memory problem. | 27 (bandit 300k steps, the others 2M) |
| Credit gap | After each update: the gain in log π(a_t) on steps followed by a reward within 8 steps (1 on the controls), minus the gain on the other steps. Positive means the update turns reward into policy change; near 0 means the policy moves regardless of reward. The standard error is taken over blocks of 100 updates. | logged in every run |
| Entropy ablation | AC+aux entropy cost 0.001 and 0, against P1's 0.01 | 9 Memory Maze runs |
| Learning-rate ablation | Disco103 at 3e-3 and 1e-2 (the upstream colab's value), against P1's 3e-4 | 9 Memory Maze runs, plus controls at 1e-2 |

Memory Maze runs used 2M steps, enough to cover P1's drift back to uniform. The main measure is the training reward: every training step is a sampled action of the current policy, so the last quarter alone covers 500k steps per run. This is far more than the 8 validation episodes.

### 3.1 Positive controls

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/results/figures/p1d_controls-dark.png">
  <img src="reports/results/figures/p1d_controls.png" alt="Reward per trial on three cue-recall controls for AC+aux at lr 3e-4 and Disco103 at lr 3e-4 and 1e-2, three seeds each">
</picture>

Reward per trial in the last quarter (chance = 1/6 = 0.17):

| Task (pass line) | AC+aux, lr 3e-4 | Disco103, lr 3e-4 | Disco103, lr 1e-2 |
| --- | --- | --- | --- |
| One-answer bandit (0.8) | **3/3** (0.98–0.99) | 0/3 (0.02–0.04) | 2/3 (0.99, 0.83; third 0.10) |
| Reactive cue (0.8) | 1/3 (0.98; others 0.67, 0.17) | 0/3 (0.18–0.19) | 0/3 (0.17) |
| Cue held for 8 steps (0.5) | 0/3 (0.16) | 0/3 (0.16) | 0/3 (0.16) |

- **AC+aux learns from reward.** It solves the bandit, and one seed solves the pixel cue task. The seed that failed collapsed onto a single cue-independent action (entropy 0.007).
- **Disco103 fails the bandit at P1's learning rate.** Its credit gap there is zero (−0.005 to −0.003, SE 0.008–0.019), so it drifts onto a wrong action regardless of reward. At 1e-2 it solves the bandit in two seeds but stays at chance on every task that needs the cue.
- **The memory control does not isolate memory.** It fails for everyone, but the reactive version already fails in most seeds.

A local check before the stage ran the **unmodified upstream DiscoRL agent** on its own Catch environment. It learns at 1e-2 and not at 3e-4 within 400 updates ([`scripts/upstream_disco_check.py`](scripts/upstream_disco_check.py)). The sensitivity to learning rate therefore belongs to the released rule, not to this code.

### 3.2 Memory Maze

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/results/figures/p1d_maze_reward_credit-dark.png">
  <img src="reports/results/figures/p1d_maze_reward_credit.png" alt="Memory Maze reward rate and credit gap for three AC+aux entropy settings and three Disco103 learning rates">
</picture>

| Condition | Reward per 1000 steps (3 seeds pooled) | vs uniform policy (95% CI) | Policy entropy, last quarter | Credit gap (10⁻³) |
| --- | --- | --- | --- | --- |
| AC+aux, entropy 0.01 (P1 setting) | 0.203 | baseline | 1.78 | +10 to +19 |
| AC+aux, entropy 1e-3 | **0.318** | **1.56 (1.35–1.81)** | 1.64–1.69 | +10 to +17 |
| AC+aux, entropy 0 | 0.264 | 1.30 (1.12–1.51) | 0.15–0.70 | +4 to +6 |
| Disco103, lr 3e-4 (P1 setting) | 0.175 | 0.86 (0.73–1.02) | 1.76–1.79 | 0.0 to +0.3 |
| Disco103, lr 3e-3 | 0.178 | 0.88 (0.74–1.03) | 1.79 | −1.9 to +0.1 |
| Disco103, lr 1e-2 | 0.175 | 0.86 (0.73–1.01) | 1.79 | −0.8 to −0.1 |

The uniform baseline is the AC+aux entropy-0.01 runs: their policy stayed uniform, with every action at 0.17. Confidence intervals pool the seeds and treat reward events as Poisson.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="reports/results/figures/p1d_maze_entropy-dark.png">
  <img src="reports/results/figures/p1d_maze_entropy.png" alt="Policy entropy over 2M steps for each Memory Maze condition, three seeds per panel">
</picture>

- **AC+aux at P1's setting has a learning signal, but its policy stays uniform.**
  - **Entropy bonus 1e-3** (10× smaller): all three seeds earn more than any seed at 0.01. Each action's frequency stays near uniform (0.11–0.24), but the per-step entropy is lower. The policy therefore seems to react to what it sees rather than repeat one action.
  - **No entropy bonus:** the policy collapses onto `forward_right` (69–95% of actions) and earns less.
- **Disco103 shows no reward credit at any learning rate.** By the end of training its policy target has the same entropy as its policy, and the policy is back to uniform. Raising the learning rate fixed the bandit but does nothing here.
- **Ignore the "vs uniform" column in [diagnostics.md](reports/results/p1d/diagnostics.md).** Its scripted references used only 32 episodes, and the uniform one earned 5 rewards (0.156 per 1000 steps). The table above uses the AC+aux entropy-0.01 runs as the uniform baseline instead.

## 4. Interpretation

1. **The pipeline works; the learners are the problem.** Reward wiring, the pixel encoder, the recurrent unroll and the update loop are enough for AC+aux to solve the bandit and, in one seed, the pixel cue task.
   - **A real bug did turn up.** On CPU, `jnp.asarray` aliased the environment's previous-action and previous-reward buffers, so stored rollouts were overwritten. It is fixed and covered by a regression test.
   - **P1 was likely unaffected.** It ran on the GPU, where these arrays are copied to the device.
2. **Disco103 cannot be the base learner at this budget.** It needs a learning rate around 1e-2 even for a bandit. At no tested learning rate does it produce reward credit from sparse reward or from pixels. A likely reason, not tested: its targets come from its own prediction heads, which must first learn to predict reward, and in Memory Maze about 4 in 5 batches contain no reward at all. P1's Disco arm therefore says nothing about memory or spatial abilities.
3. **AC+aux's failure in P1 came from a hyperparameter.** Its entropy bonus of 0.01 outweighed the reward signal in this sparse task. Lowering it gives a reliable but small gain, of the size a reactive policy (turning at walls) can reach.
4. **The budget is far too small for the question.** 2M–5M steps means 2,000–5,000 updates. The [Memory Maze repository](https://github.com/jurgisp/memory-maze) shows an IMPALA agent trained for 400M steps, and humans average 26.4 rewards per 9×9 episode, against 0.2 for random play. Spatial-memory abilities cannot form before the agent collects enough rewards to learn from.

## 5. Limitations

- Three seeds per condition, so every result is descriptive. GPU runs are not bitwise deterministic.
- One maze size (9×9) and one network size.
- The credit gap measures one update's effect on the actions it was computed from. It is a diagnostic, not a causal test of learning.
- The rate-ratio CIs pool seeds; the seed-to-seed spread is shown separately in the figures.
- `best_validation_return` is the maximum of periodic 8-episode validations (greedy in P1, sampled in P1d), so it is noisy and biased upwards. It was used only to pick checkpoints, never as a result.
- The replay, self-imitation and teacher conditions (H2, H3, P2) never ran at scale. The only teacher evidence is that it learned nothing without successes.

## 6. Next steps if the study resumes

1. **Make the base learner pass first.** Use AC+aux with entropy cost 1e-3 and more updates per collected step (a higher replay ratio or smaller batches). Require the reactive cue control to pass in 3/3 seeds as a gate.
2. **Raise throughput.** Each run steps at about 223 transitions/s, limited by CPU (Mesa) rendering. NVIDIA EGL with one context per worker used too much GPU memory, so a shared or batched GPU renderer is the main cost lever.
3. **Treat Disco103 as a question of its own.** Either test it closer to its meta-training regime (denser reward, its own learning rate), or report the negative result together with the credit-gap diagnostic.
4. **Only then run P2:** the pilot matrix (5 seeds; replay, self-imitation and teacher conditions) and a power plan.

## 7. Reproducing the numbers and figures

Raw results are gitignored under `runs/`. The summary numbers, stage reports and figures in `reports/results/` are tracked. P1d ran the code of commit `684aa7b`. The server recorded it as `a71f68e`, its hash before the old data folders were dropped from the history; the code is identical.

```bash
python scripts/expand_conditions.py --stage P1d_diagnostics --output runs/diagnostics/configs
bash scripts/run_stage.sh runs/diagnostics/configs 4 8
python scripts/diagnose_stage.py runs/diagnostics --references --parallel 8
python scripts/plot_results.py
```

`plot_results.py` reads training logs only. It expects the two copied archives at `runs/pilot_results/` and `runs/diagnostics_results/`; flags change the paths.

- **Inputs.** Each run writes `metrics.jsonl`, one line per update with policy entropy, action counts and reward events in the collected chunk. `diagnose_stage.py` adds a per-run summary, `diagnostics.json`, with last-quarter reward, credit gaps with block standard errors, and entropy.
- **Reward rate.** Reward events divided by environment steps, × 1000, taken over the last quarter of training or over bins of 250 updates (P1).
- **Entropy curves.** Mean policy entropy over bins of 100 (P1) or 20 (P1d) updates.
- **Baseline and ratios.** The uniform baseline and the rate ratios are computed as in §3.2. The scripted forward-left line is drawn but marked as rough.
- **Rendering.** Matplotlib, one colour per learner (blue AC+aux, orange Disco103), seeds as separate lines or dots. Each figure is saved in a light and a dark version, and the page shows the one that matches the reader's theme.

## 8. What is implemented

| Item | Status |
| --- | --- |
| Memory Maze RGB-only wrapper with a separate evaluation sidecar | Real reset/step checked |
| Render telemetry (segmentation visibility, depth-backprojected known cells) | Re-rendered RGB matches the observation pixel for pixel; 99.8% of floor and 99.7% of wall pixels land in the right cells |
| CNN + previous action/reward + GRU with sequential unroll | Online/unroll, reset and history-gradient checks |
| Disco103, AC, AC + next-policy auxiliary | Learner checked with the real frozen rule; upstream agent reproduced on Catch |
| Fresh, recent sequence replay, positive-advantage self-imitation | Smoke runs on the real benchmark |
| SDPO-style self-teacher (outcome / outcome + success memory, shuffled control, evidence gate, teacher acting) | Implemented; usefulness never shown (P1: Δ = 0) |
| Six-family ability suite, navigation with memory interventions, cross-seed report | Ran on P1 checkpoints |
| Synthetic cue-recall controls, credit diagnostics, scripted references (`diagnose_stage.py`) | Ran in P1d |
| Tests | 94 pass (`python -m pytest -q`) |

## 9. Running the code

Use Python **3.12**. The stack below was used for local macOS CPU checks; Linux CUDA and EGL are set up separately on the server.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m disco_spatial --help
MUJOCO_GL=glfw python -m disco_spatial doctor --environment --output runs/environment/doctor.json
```

Memory Maze 1.0.3 needs Gym at import even on the dm_env path, so Gym is pinned; versions are pinned in `pyproject.toml`. Its border drawing covers the whole image below 64 pixels, so at least 64 pixels are enforced.

**GPU server** (Linux, NVIDIA):

```bash
bash scripts/setup_gpu.sh
cp scripts/server_env.example.sh server_env.sh && . ./server_env.sh
python scripts/expand_conditions.py --stage P1_learning_gate --output runs/pilot/configs
bash scripts/run_stage.sh runs/pilot/configs 4 8
python3 scripts/stage_status.py runs/pilot
python scripts/evaluate_stage.py runs/pilot --parallel 4 --cores 8
```

Lessons from the Vast.ai L40S container:

- **Speed.** Environment stepping sets the speed. `env_processes=true` gives each environment its own worker process, which raised one run from 23 to about 200 steps per second.
- **Threads.** XLA and OpenBLAS create a thread per visible host core (255). `server_env.sh` sets BLAS threads to 1, and `run_stage.sh` gives each run its own block of cores with `taskset`.
- **Renderer.** One NVIDIA EGL context takes about 650 MiB of GPU memory, so all server runs use Mesa EGL. The two renderers differ at edge pixels and are never mixed within a study.

**Single runs and evaluation.**

```bash
python -m disco_spatial train --config configs/integration.json --output runs/integration/disco-fresh
python -m disco_spatial train --config configs/smoke.json --teacher outcome_memory \
  --teacher-query counterfactual_success --distillation-weight 0.5 --output runs/smoke/disco-fd-memory
python -m disco_spatial navigate --checkpoint runs/pilot/disco-fresh-seed0/best.pkl \
  --seeds 2000000 2000001 --interventions none reset_every_step truncate_16 --output runs/navigation/disco.json
python -m disco_spatial scripted --policy uniform --seeds 2000000 2000001 --output runs/scripted/uniform.json
```

Seeds are split into namespaces: training below 1000, validation 1,000,000+, policy test 2,000,000+, and probe partitions 3,000,000+ to 5,000,000+. Existing output directories are never overwritten. Exact resumption of simulator and replay state is not supported. The ability suite (`collect`, `extract`, `abilities`, `report`) and every convention are described in [METHODS](docs/METHODS.md) and the [research protocol](docs/RESEARCH_PROTOCOL.md). The earlier TripoSplat and object-recognition experiments are in the [legacy archive](docs/LEGACY_ARCHIVE.md).

## Sources

- [Memory Maze](https://github.com/jurgisp/memory-maze), [paper](https://arxiv.org/abs/2210.13383)
- [DiscoRL](https://github.com/google-deepmind/disco_rl), [pinned source and checkpoint provenance](third_party/PROVENANCE.json)
- [SDPO](https://arxiv.org/abs/2601.20802), [official code](https://github.com/lasgroup/SDPO)

Last update: 2026-10-31.
