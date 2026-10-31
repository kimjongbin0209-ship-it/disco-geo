# Spatial abilities under limited reward and SDPO-style hindsight-feedback distillation: research design

**English** | [한국어](SDPO_SPATIAL_RESEARCH_PLAN_ko.md)

## Research question

**With the same limited reward and interaction budget, which spatial abilities does Disco form, and how do experience reuse and hindsight feedback change them?**

The TripoSplat-based object experiments were removed from the main study ([legacy archive](LEGACY_ARCHIVE.md)). This document covers the development plan for the ability evaluation and the RL learning rule. Statistics, splits and leakage rules follow the [research protocol](RESEARCH_PROTOCOL.md); implementation details follow [METHODS](METHODS.md).

## Benchmarks

| Role | Environment | Reason |
| --- | --- | --- |
| Main study | Memory Maze 9×9 RGB | +1 each time the requested object is touched, then a new target is requested. The maze and object positions stay fixed within an episode, so memory, position tracking, map formation and revisits can be studied together. |
| Follow-up validation | RoboTHOR RGB ObjectNav | Checks generalization in realistic scenes with terminal success and SPL. Not mixed with pretrained features or geodesic shaping. |

Memory Maze does not give one terminal reward per episode; it gives one kind of reward each time a target is reached. The main condition uses the upstream reward unchanged, with no step cost or distance shaping.

## Ability families and their measurement

Evaluation labels are built only from the sidecars of evaluation splits and never enter the learner's inputs. Visibility and the known map come from re-rendering the actor camera at the same pose: segmentation counts target pixels and back-projected depth gives the observed cells. The re-rendered RGB matches the observation pixel for pixel. On real episodes, 99.8% of floor pixels landed in free cells and 99.7% of wall pixels in wall cells, and every visible target's bearing was within the ±40° field of view plus the sphere radius.

| Family | Measurement | Implementation |
| --- | --- | --- |
| Self-localization and motion | Position and heading relative to the start pose, relative motion over 1 and 8 steps, drift by time bin | `suite.self_localization` |
| Spatial memory | **Egocentric location RMSE of targets that were seen but are now hidden** (primary 1) | Causally accumulated segmentation-visibility mask |
| Place recognition | Nearest-neighbor recall@1 for the same place and heading at least 50 steps apart, with the chance rate | Cosine retrieval, no training |
| Maps and spatial relations | **Balanced wall/free IoU over cells observed so far** (primary 2), with unobserved cells scored separately | Canvas aligned to the start cell and the grid axis nearest the start heading |
| Local geometry and affordances | Rendered 8×8 depth RMSE, ±30° free-space distances, forward-traversability AP | Single-frame image features |
| Actual spatial behavior | Return, **matched revisit efficiency gain** (primary 3), memory-reset and history-truncation interventions | Frozen-policy test episodes |

New layouts, longer horizons, larger mazes and visual changes are not separate abilities; they are generalization axes of each ability.

### Three levels of evidence

1. **Information exists:** a small linear probe reads it from the frozen belief. Controls are frame-only image features, an RGB-free action history, the untrained `init.pkl` with the same initialization, and the train mean.
2. **It transfers:** a new controller trained on the frozen representation uses it. This involves extra supervision, so it is not zero-shot.
3. **It is an actual ability:** behavior improves with the actor itself frozen; revisit efficiency rises and performance drops under a memory reset.

An improved map probe alone does not show that a planning ability has formed.

## What we take from SDPO

The core of [SDPO](https://arxiv.org/abs/2601.20802) is not repeated imitation of past successes. It uses the distribution of a policy that has seen feedback as the learning signal. The elements confirmed in the original paper and their counterparts here:

| SDPO (LLM) | This study (recurrent RGB policy) |
| --- | --- |
| The current model with feedback in its context is the teacher | A self-teacher: a feedback head on the student's GRU belief (stop-gradient) |
| Loss KL(student ‖ stopgrad(teacher)) per token | KL(student ‖ stopgrad(teacher)) per action step, over the full 6-action distribution |
| Regularized by an EMA teacher (α=0.01); diverges without regularization | Targets come from an EMA of the teacher head (`teacher_ema_rate=0.01`) |
| With scalar rewards, another successful rollout of the same problem is the feedback (never the attempt itself) | The earlier success frame for the same target color in the same episode (`success_memory`, causal, never the attempt itself) |
| Feedback is hindsight information about the attempt | Whether a reward arrived within H steps of the attempt (`outcome`, hindsight about the agent's own attempt) |
| Experiments are strictly on-policy | Each latest chunk is used exactly once, right after its whole outcome window is observed (`MaturedChunks`, age logged) |

Memory Maze builds a new maze at every reset, so "the same task instance" is one episode and one target color. This is the design answer to the logical-instance and attempt-ID problem.

### Why the self-teacher must be trained, and its structure

An LLM teacher can already interpret feedback; a CNN/GRU trained from scratch cannot. The feedback pathway is therefore trained explicitly.

```text
teacher_logits = base_logits + Δ_ψ(stopgrad(belief_t), feedback_t)
training: behavior cloning of actual actions with behavior logits as base (conditioned on observed outcomes)
target:   stopgrad(student logits) + Δ_EMA(belief_t, query feedback)
```

By Bayes' rule, the optimal Δ is log P(feedback | history, action), up to a per-step constant. Three consequences follow.

1. **Without conditioning on outcomes the teacher learns nothing.** On on-policy data, actions are conditionally independent of any causal information outside the history, so a teacher given only past success frames converges to Δ=0. Every teacher therefore has an `outcome` channel, and `success_memory` is meaningful only together with the outcome.
2. **Querying success yields a soft policy improvement.** The target is proportional to π(a|h)·P(success | h, a). A unit test recovers this posterior within 0.04 (`tests/test_teacher.py`).
3. **Its expectation equals self-imitation.** On successful steps the expectation of this target equals that of success-conditioned imitation (SIL), so SDPO-style distillation is the Rao–Blackwellized form of SIL. The **prediction for H2** is therefore: if FD beats SIL, the reason must be lower variance and teacher generalization, not extra reward information. If they are equal, that is consistent with the theory.

There are two queries. `actual_success` distills only on steps that actually succeeded; it is hindsight, so the teacher cannot act. `counterfactual_success` queries "success soon" on every step. Because the query is constant and the memory is causal, this teacher **can act**, so it can be compared directly with the student on the same validation episodes (`evaluate --teacher`).

### The two places past rollouts are used

| Mode | Role of past rollouts | Implementation |
| --- | --- | --- |
| Feedback memory | Reference material for the teacher while learning from the latest attempt | `success_memory` channel: the earlier success frame in the same episode |
| Sequence replay | Losses and gradients are computed again on past rollouts themselves | `recipe=recent_replay`: original behavior logits, prefix reconstructed with current parameters |

The full loss is:

```text
L = L_Disco,fresh [+ α·L_Disco,replay] + λ_SIL·L_SIL + λ_T·L_teacher-BC + λ_SD·g·KL(student ‖ sg(teacher_EMA))
```

g is the evidence gate below. This loss is reported as an extension of the released Disco103. The rule's internal state (meta-RNN, advantage EMA, target EMA) is updated from fresh data only.

## How teacher usefulness is verified

Diagnostics are computed every update on chunks the teacher has not been trained on yet, so they are held out. The gate opens only when the accumulated diagnostics satisfy all four conditions.

| Diagnostic | Meaning | Gate condition |
| --- | --- | --- |
| Information gain (success steps) | E[log teacher(a) − log μ(a)]: how much the feedback explains successful actions | ≥ `teacher_min_information_gain` |
| Shuffled information gain | The same with feedback shuffled across steps | Margin to the gain ≥ the same threshold |
| Local improvement | E[(teacher(a)/μ(a) − 1)·G^H]: a first-order estimate of policy improvement | > 0 |
| Copy rate | Fraction of steps where the teacher gives the taken action ≥ 0.95: detects action copying | ≤ `teacher_max_copy_rate` |

The gate closes again if the conditions deteriorate, and its open/close history is recorded in `completed.json`. In addition, the causal teacher (success query plus memory) acts on validation episodes and is compared with the student (G4b). `teacher_gate=never` trains and validates the teacher without distilling; `always` is an ablation that forces the gate open.

The teacher's inputs are only `own_reward_outcome_within_horizon`, `own_rgb_frame_of_previous_same_color_success`, `elapsed_steps` and `stopgrad_student_belief`, and the protocol validator enforces this. A teacher given GT maps, poses, depth or shortest paths would be extra spatial supervision; such conditions belong to a separate guided study only (A4, not implemented).

## Comparison conditions and interpretation rules

The condition table is in [`experiments/conditions.json`](../experiments/conditions.json), and `scripts/expand_conditions.py` expands it into per-seed configs. Every condition is applied to both Disco and AC+aux (same recurrent backbone, same per-seed initialization).

| Condition | What changes | Kind of signal |
| --- | --- | --- |
| fresh | Nothing | Baseline |
| replay | Rule loss on past sequences | Reuse of the same reward |
| self_imitation | Positive-advantage imitation of successful chunks | Objective change, same reward |
| fd_outcome | Distillation from the outcome teacher | Hindsight-feedback pathway built from the same reward |
| fd_memory | Outcome plus earlier success frame, success query | Feedback memory |
| fd_outcome_shuffled | Same compute and masks, feedback pairing destroyed | Control for feedback information |
| fd_outcome_replay | Distillation combined with sequence replay | Both axes combined |
| teacher_only | The teacher is trained and validated only | Teacher validation |

"Better use of the same reward" and "an additional learning signal" are labeled separately. No observation-based auxiliary objective has been added yet: for example, action-conditioned prediction would be a new self-supervised objective and a coverage bonus would change the reward; such conditions will be marked as separate axes. Along with environment interactions we report `learner_network_steps` (an approximate count of per-observation network evaluations by the learner) and wall time, and compare in two panels: sample budget and compute budget.

## Development order and execution gates

1. Memory Maze integration, real reset/step and render telemetry checks — **done**
2. Recurrent training path: block-wise prefix reconstruction, vectorized replay, `init.pkl` — **done**
3. Common ability evaluation: telemetry collector, six-family suite, navigation, cross-seed report — **implemented and smoke-checked; not yet run at research scale**
4. Replay and self-imitation baselines — **implemented, smoke-checked**
5. Self-teacher validation: held-out diagnostics, shuffled control, gate, teacher acting — **implemented; the offline pre-check with a fixed explore policy showed no information gain (below); usefulness during RL training not verified**
6. SDPO-style combinations: fresh with lookahead, combination with replay, memory channel — **implemented, smoke-checked**

`P1_learning_gate` ran on a GPU server (Vast.ai L40S) from 2026-10-28: 12 runs of 5M steps (Disco and AC+aux × fresh and teacher_only × 3 seeds). **G2 was not passed.** Every run's policy entropy reached ln 6 ≈ 1.792, a uniform policy, within roughly the first 0.5–1M steps, and the training reward stayed at the uniform-policy rate (about 0.2 per 1000-step episode). One run, while biased towards `forward_left`, briefly doubled its reward rate and then drifted back to uniform. Test returns (8 episodes per run) did not differ from random. The teacher's information gain equalled its shuffled control to 1e-9, so its Δ head stayed at zero (G4a not passed). Frozen-probe abilities after training were no better than, and in places worse than, the untrained init (loop-closure recall 0.38 → 0.31–0.33). Pilot evidence only; results are in `runs/pilot_results`.

P2 waits until the learners learn. `P1d_diagnostics` asks why P1 failed G2, with three checks:

* **Positive controls:** `cue_recall` is a synthetic task in which the cue colour names the rewarded action, either immediately (`control_reactive`) or after 8 blank steps (`control_memory`). A fixed cue turns it into a one-answer bandit (`control_bandit`) that needs no perception at all. Both learners must clear these before any Memory Maze result is interpreted. The task is restricted to the development purpose and is never a study task.
* **Credit diagnostics** (`credit_window=8`): each update compares how much it raises the taken action's log-probability on steps followed by a reward against steps that are not. The rule's own signal is logged too: the target log-probability gain for Disco, the advantage for AC.
* **Entropy ablation:** AC+aux is run with entropy cost 0.001 and 0, against P1's 0.01.
* **Disco103 learning rate:** Disco is run at 3e-3 and 1e-2 (the upstream evaluation colab's value), against P1's 3e-4, on Memory Maze and on the controls.

Local CPU checks before the stage used a small network (channels 16/32, hidden 64) and one seed each, so they are development evidence only.

* **Bandit:** AC+aux solved the bandit in about 100 updates at 3e-4. Disco103 at 3e-4 moved *away* from the answer within 200 updates. At 3e-3 it sat on a uniform plateau (entropy 1.792) for about 250 updates and then solved the bandit; at 1e-2 it solved it by about 300 updates.
* **Upstream harness:** the unmodified upstream Agent behaves the same way. On a bandit at 3e-4 it collapses onto a wrong action. On upstream Catch, 3e-4 shows no learning in 400 updates, while 1e-2 starts learning after about 300 updates (`scripts/upstream_disco_check.py`). The released rule therefore works in this software stack, but only in a learning-rate regime far from P1's.
* **Visual cue task** (`control_reactive`, 300k steps): neither learner solved it. AC+aux reached 0.21–0.23 reward per step (chance 0.17, optimum 1). Disco at 3e-3 and 1e-2 converged on one cue-independent action, and the spread of its encoder features across cues collapsed from 3.6 to 0.25. The untrained network does separate the cues (belief spread 1.45), so the information is lost during Disco training, not missing at the start.

The positive-control budgets were raised accordingly (`control_reactive` to 2M steps). Their credit window is 1, because the scored action is rewarded on the next step.

The stage uses 2M steps (P1's collapse happened within that budget) and sampled validation. `scripts/diagnose_stage.py` then scores on-policy training reward against scripted references (uniform, and `forward_left`/`forward` biases) on the same test seeds. Test evaluation itself is widened to 32 or more episodes, with sampled actions matching the random baseline (`evaluate_stage.py --test-seeds 32 --policy sampled`).

```bash
python scripts/expand_conditions.py --stage P1d_diagnostics --output runs/diagnostics/configs
bash scripts/run_stage.sh runs/diagnostics/configs 4 8
python scripts/diagnose_stage.py runs/diagnostics --references --parallel 8
python scripts/expand_conditions.py --stage P1_learning_gate --output runs/pilot/configs
python -m disco_spatial train --config runs/pilot/configs/disco-fresh-seed0.json
python -m disco_spatial collect --partition probe_train --seed 3000000 --episodes 32 --output datasets/probe_train
python -m disco_spatial extract --checkpoint runs/pilot/disco-fresh-seed0/best.pkl \
  --collection datasets/probe_train --output runs/features/disco-fresh-seed0/probe_train
python -m disco_spatial abilities --train ... --validation ... --test ... --output runs/abilities/disco-fresh-seed0.json
python -m disco_spatial navigate --checkpoint runs/pilot/disco-fresh-seed0/best.pkl \
  --seeds 2000000 2000001 2000002 2000003 --interventions none reset_every_step truncate_16 \
  --output runs/navigation/disco-fresh-seed0.json
python -m disco_spatial report --reference fresh \
  --runs fresh:0:runs/abilities/disco-fresh-seed0.json:runs/navigation/disco-fresh-seed0.json ... \
  --output runs/reports/pilot.json
```

**P1d results (2026-10-31; 45 runs, 3 seeds per condition).** G2 is still not met, and the study is paused here. The full report with figures is in the [README](../README.md#3-p1d-why-did-it-fail); the numbers are in [`reports/results/summary.json`](../reports/results/summary.json).

* **Positive controls:** AC+aux passed the bandit in 3/3 seeds and the reactive cue task in 1/3 (0.98 per trial; the others reached 0.67 and 0.17). Disco103 failed the bandit at 3e-4 (0/3, credit gap 0) and passed it in 2/3 seeds at 1e-2, but stayed at chance on the reactive and memory tasks at both learning rates. No run passed the 8-step memory control.
* **Memory Maze, Disco103:** the credit gap was 0 at 3e-4, 3e-3 and 1e-2, and the policy returned to uniform (0.175–0.178 rewards per 1000 steps, against 0.203 for a uniform policy).
* **Memory Maze, AC+aux:** the credit gap was positive in all 9 runs (+4 to +19 ×10⁻³). Entropy cost 0.01 kept the policy uniform; 1e-3 raised the reward rate to 0.318 (1.56×, 95% CI 1.35–1.81); 0 collapsed the policy onto `forward_right` (0.264).
* The scripted references used only 32 episodes, so the uniform baseline is the pooled last quarter of the AC+aux entropy-0.01 runs instead.

## What has been run so far (not research results)

The summary is in [`reports/development/smoke_summary.json`](../reports/development/smoke_summary.json); the raw data is in the gitignored `runs/smoke`, `runs/teacher_validation` and `datasets/smoke`.

- **Real Memory Maze smoke matrix.** Seven conditions were trained for 2,048 steps each (fresh, AC+aux, replay, self-imitation, outcome teacher, memory teacher with the gate forced open, shuffled control). All ended with finite losses. No reward was earned during training, so self-imitation found no reward chunk and the evidence gate stayed closed (zero distillation updates); the success paths are verified by unit tests only. `disco-sil` was bit-identical to `disco-fresh` because no auxiliary batch was formed, which confirms shared initialization and determinism.
- **Ability suite and navigation.** Feature extraction, the suite, navigation and the report ran end to end on real data from a telemetry collection (4 train, 2 validation, 2 test episodes, all different layouts). The smoke checkpoints' seen-hidden RMSE was at the train-mean level (about 4.3 cells), as expected after such short training. However, **untrained `init.pkl` features alone** gave place-retrieval recall of 0.28 (chance 0.055) and forward-traversability AP of 0.94 (base rate 0.19). These two families must therefore be compared against the untrained control, not chance.
- **Offline teacher pre-check.** 60 real episodes were used, with a sticky explore behavior policy that ignores observations, 24 rewards and 384 held-out success steps. None of five teacher variants (hidden 128/16/linear, with and without L2) reached positive held-out information gain. Unregularized heads overfit and ended below the shuffled control; L2-regularized heads converged to Δ≈0. The gate stays closed in every case. This confirms on real data the concern that "putting past rollouts in the context does not by itself produce a good target." The teacher can become useful only once a trained policy succeeds often enough, so the pilot config sets `teacher_l2=0.001` and the `teacher_only` stage measures when information appears.

## Known risks and limitations

- **Early successes are rare.** Early in training successes are rare, so the teacher's BC signal is weak and the gate may stay closed for a long time. This is intended behavior, not a defect. Gate evidence is weighted by success steps and requires at least `teacher_min_success_steps`. The `teacher_only` condition measures separately when information appears; in the offline check above it did not.
- **SDPO-style distillation may bring no benefit.** In theory its expectation equals self-imitation; if the teacher does not generalize there may be no gain, which is also a valid H2 result.
- **Matured chunks lag by H steps.** Their age is about (L+H)/L updates. This is not strictly on-policy, so it is logged as `feedback_chunk_age`.
- **The teacher may not be causal.** The `actual_success` query is hindsight, so the teacher cannot act; acting validation uses only the counterfactual success query.
- **Self-imitation changes the objective.** Its internal return is an n-step bootstrapped return, not an off-policy-corrected policy gradient.
- **Exact resumption of simulator and replay state is not supported.** Run in a new output directory.
- **The local macOS CPU does about 100 transitions/s.** Research-scale training must be checked separately on the Linux GPU host.
