# Implementation conventions and verification criteria

**English** | [한국어](METHODS_ko.md)

## Collection and recurrent state

The environment provides only the original Memory Maze RGB and the 0/+1 reward. The target color is already part of the image border. The previous action and reward are causal information that has already occurred. At FIRST the hidden state and the previous action/reward are reset. The actor GRU, the action-conditioned prediction LSTM and the frozen update-rule LSTM are distinct.

The collector counts only real environment actions as transitions. If any member of the batch reaches LAST, the chunk ends and that observation is used only for bootstrapping. Only the finished environments are reset before the next chunk. The hidden state computed at the bootstrap is not written back into the collector carry, so the next chunk's observation is not consumed twice. A time-limit LAST (continuation = 1) is distinguished from a true termination (continuation = 0).

The online carry is the collector's actual truncated-BPTT state. We do not claim it is identical to re-running the whole episode prefix with new policy parameters each time. Replay re-runs the full raw prefix with the current parameters to obtain the state at the start of the loss window.

## Recent replay

The replay sampler accepts only complete episodes or immutable raw prefixes that extend from FIRST. T actions correspond to T+1 RGB frames. Full behavior logits, original rewards, policy versions, goal counters and termination/truncation flags are preserved. Evaluation labels live in a separate sidecar.

From the same pre-update parameters and a copied meta-state, the fresh and replay losses are each normalized and combined as `(1−alpha)grad_fresh + alpha grad_replay` in one update. The meta-RNN, advantage/TD EMAs and target-network EMA commit only the fresh branch's changes. alpha = 0 is identical to the baseline update. This fresh-only lifetime-state convention is an added training strategy.

The development defaults are a maximum replay age of 20 updates and an absolute log-ratio limit of 5 for the selected action. These are development settings to be checked, not optimal values. Excessive drift or non-finite diagnostics fall back to a fresh update. The reported rho is clipped to log-ratio ±20 for numerical stability, but the learner receives the original behavior logits. There is currently no explicit KL/ESS threshold gate; those values are logged.

AC uses V-trace and Disco's Q uses a Retrace-style correction. This alone does not guarantee that all of the frozen rule's policy/y/z targets are unbiased or stable on old replay. Reusing raw experience is distinguished from reusing the features or meta-targets computed at the time.

With `env_processes=true`, each environment runs in its own worker process (spawn). The batch issues every command first and then collects the results. Worker timesteps and bookkeeping attributes are identical to in-process execution, and a test checks that the training logs match. Worker errors reach the training process together with their traceback. Prefix reconstruction at the same episode position (the Disco EMA target state and feedback chunks) is computed once on the batch axis.

Prefix reconstruction unrolls in blocks of 256/64/16/4/1. Compilation count is bounded and the recurrence equals step-by-step processing (checked at atol 2e-5 over 300 steps). Replay eligibility is vectorized: since policy versions are monotonic within an episode, a window's oldest version is at its start and its newest at its end.

## Self-imitation

With `self_imitation_weight > 0`, each update samples one reward-containing chunk from replay (success selection, within `self_imitation_max_age`) and adds a positive-advantage imitation loss on it. The return is the n-step return from the chunk's real rewards, bootstrapped with the current value at the chunk's end. The baseline V is the rule's own categorical value (Σπq for Disco, v for AC), with a stop-gradient. This changes the objective; it is not an off-policy-corrected policy gradient.

## SDPO-style feedback self-teacher

The design rationale is in the [research design](SDPO_SPATIAL_RESEARCH_PLAN.md). The implementation conventions are:

- **Teacher.** The teacher is a residual head with `teacher_*` parameters. Its logits are `base + Δ(stopgrad(belief), feedback)`. The output layer is zero-initialized, so the initial teacher equals its base. Backbone parameters, optimizer moments and meta-state are initialized from the same RNG splits as a teacher-free agent (tested).
- **Feedback.** `outcome` indicates whether a +1 reward arrives within H steps. Unobserved windows are masked as censored; a shortfall in a finished episode counts as a real failure. `success_memory` is the frame from the previous success (the approach view) on the same color as the current target, read from the RGB border, plus the elapsed steps. It becomes available from the observation after the success and is causal.
- **Data.** `MaturedChunks` releases each environment's latest chunk exactly once, after its whole outcome window has been observed. Waiting for the whole window rather than the first reward avoids releasing successes earlier than failures. Prefix states are reconstructed with the current parameters; the original behavior logits are used unchanged.
- **Losses.** Teacher BC is computed on known steps with the behavior logits as base (λ_T). Distillation is `KL(student ‖ stopgrad(stopgrad(student) + Δ_EMA(query)))` (λ_SD·gate). `actual_success` distills only on steps that actually succeeded and is hindsight. `counterfactual_success` fixes the query outcome to 1 and distills on every step; it is causal. The shuffle control permutes only the teacher inputs across (time, batch); masks and returns stay with their steps.
- **One update.** The rule gradient (fresh, or mixed with replay) and the auxiliary gradient are added and the optimizer is applied once. The rule meta-state is updated from fresh data only. Without auxiliary batches the step is bit-identical to `learner_step_mixed`.
- **Gate and validation.** `TeacherGate` accumulates held-out diagnostics weighted by the number of steps each describes: information gain and shuffled gain by success steps, local improvement and copy rate by known steps, decayed by 0.98 per update. Distillation opens when the effective number of success steps is at least `teacher_min_success_steps` and all four criteria pass. Chunks without any success do not dilute the success evidence. At every evaluation, the causal teacher's own result when acting on the same validation seeds (`teacher_validation`) is recorded as well.
- **Capacity and regularization.** With `teacher_hidden=0`, Δ is linear in [belief, feedback] and their outcome interaction. `teacher_l2` adds an L2 penalty on the head parameters. In an offline check on real explore episodes (`scripts/validate_teacher_offline.py`), unregularized heads ended below the shuffled control on held-out data, and L2-regularized heads converged to Δ≈0. The pilot default is therefore `teacher_l2=0.001`.

The teacher's inputs contain no simulator state, map, pose or target position. The protocol validator enforces the teacher-input whitelist and `validated=false`. `FeedbackMemory` (guarded storage) remains as a legacy API; the training path uses `disco_spatial.feedback`.

## Evaluation and research gates

There are three primary endpoints: seen-but-hidden target location, the causally known wall map and matched revisit efficiency. With `telemetry=True`, the evaluation environment re-renders the actor camera: it counts target pixels with segmentation, finds observed cells by back-projecting depth and records an 8×8 depth map. The same renderer's RGB matches the observation pixel for pixel. The doctor and collection manifests record the fraction of floor pixels landing in wall cells as an oracle self-check. The ability suite (`abilities`), navigation (`navigate`) and cross-seed aggregation (`report`) are implemented but have not yet been run at research scale.

Generic feature extraction never passes labels to the model. Image features and GRU beliefs are extracted from the causal raw history, and evaluation labels are joined afterwards in readout-only artifacts. The original learner trajectory schema does not accept such artifacts as inputs. Position labels are fixed to the episode's start position and heading and are never aligned using test labels.

Checkpoints are selected by validation return on the original task. We never pick the best test checkpoint per ability. Policy performance is distinguished from frozen-readout performance, and the common-history distribution from trajectories generated by each policy. The full statistical conventions follow the [research protocol](RESEARCH_PROTOCOL.md).

## Not yet verified

- Sufficient learning on the new benchmark and a causal performance benefit from memory
- Research-scale measurement of all six ability families and the RL-train geometry inventory
- Actual usefulness of the SDPO teacher: whether the gate opens and whether the teacher acting beats the student
- Long-run stability, VRAM and budget on the Linux GPU host (throughput has been measured; see the README)
- Exact resume including simulator and replay state

Small integration results are evidence of code wiring, not of emergent spatial abilities.
