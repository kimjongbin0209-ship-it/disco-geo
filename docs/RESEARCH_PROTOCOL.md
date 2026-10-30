# Research protocol for measuring the emergence of spatial abilities under limited reward

**English** | [한국어](RESEARCH_PROTOCOL_ko.md)

## Research question and scope of claims

After training on a single RGB navigation task with a limited success reward,
we measure which spatial information appears in the frozen visual features and
recurrent state. This is not a study that builds a multi-task supervised learner
teaching every ability at once.

Three hypotheses are tested in order.

- **H1 — update rule:** With the same CNN, memory, initialization and task, does
  Disco improve spatial representations over actor-critic + next-policy auxiliary?
- **H2 — experience reuse:** Beyond the effect of reusing the existing rewards of
  past trajectories, does a distillation loss from a teacher built on those
  trajectories add value?
- **H3 — information in feedback:** When a working teacher exists, how do feedback
  built from the student's own RGB/action/reward and feedback that supplies new
  privileged information differ in effect?

The H1 base learner never receives depth, pose, coordinates, maps, target vectors,
shortest paths or expert actions as inputs or loss targets. Ground truth inside the
environment that scores task success is allowed, but passing it on as per-step
distance shaping or as a teacher's answer makes a different experiment. The task
goal's color or name is task supervision.

**Calling something SDPO does not validate the teacher.**
A self-teacher adapting the [original SDPO paper](https://arxiv.org/abs/2601.20802) to the
spatial task is implemented ([design](SDPO_SPATIAL_RESEARCH_PLAN.md)), but the manifest
declares `validated=false`. Distillation is applied only after a held-out evidence gate
passes in each run. Nothing is called an end-to-end SDPO result before all of the following
are confirmed: teacher input provenance, information above the shuffled control, an actual
behavioral advantage, the distillation gradient, replay and self-imitation controls, and a
compute control.

## Boundary between existing benchmarks and our own diagnostics

For studying spatial memory itself, [Memory Maze](https://github.com/jurgisp/memory-maze)
is the direct starting point. In the standard task the maze and target positions are fixed
within an episode; the agent receives +1 each time it touches the target named by color and
then looks for the next target. This is a sparse task-success reward, but **it is not the same
as an episode-terminal-only reward**. The main condition keeps the upstream +1 on target contact
and 0 otherwise, and **adds no step cost**. The success and termination rules are not changed
either. Studying a different reward requires declaring a separate protocol variant whose results
are not mixed with the main condition.

To keep a strict terminal success plus step cost, the existing RGB-only task of
[RoboTHOR ObjectNav](https://github.com/allenai/robothor-challenge) can serve as a separate
study, with its task definition, build, episodes and success/SPL evaluator fixed. The RGB
baseline of the [official AllenAct implementation](https://allenact.org/projects/objectnav_baselines/)
also uses memory, but its pretrained features or geodesic shaping are not imported into the
geometry-free random-init experiments.

Memory Maze's online return and wall/object-location probing concepts, and RoboTHOR's
success/SPL, rest on official evaluations. The observability split, balanced map IoU, matched
revisit gain, relative pose, loop closure and reachability readouts below are **additional
diagnostics defined by this study**. Not every metric is presented as an official benchmark score.

## Ability families and fixed primary endpoints

| Family | Evaluated quantity | Status and nature of evaluation |
| --- | --- | --- |
| Local geometry | Rendered depth, ±30° free space, forward traversability | Additional diagnostic; image-feature readout implemented |
| Self-localization | Relative motion and heading, episode-origin position, drift | Additional diagnostic; implemented with an action-history control |
| Spatial memory | Location of targets seen before but not visible now | **Primary 1:** `seen_hidden_target_location_rmse` (segmentation visibility) |
| Place recognition / loop closure | Retrieval of the same place after a long time gap | Additional diagnostic; training-free recall@1 with the chance rate |
| Map topology | Observed wall map, unobserved cells separated | **Primary 2:** `known_wall_balanced_iou` (depth-backprojected known mask); reachability is follow-up work |
| Navigation | Behavior and efficiency on the original task, target revisits, memory interventions | **Primary 3:** `matched_revisit_efficiency_gain` |

Correlated measures such as depth and free space, or maps and reachability, are not each
counted as an independent "new ability." OOD is not a seventh ability but a generalization axis
of each family. The collector, suite, navigation and seed report for all six families are
implemented and have run on the real benchmark at smoke scale, but not from research-scale
training checkpoints to a final report. The manifest keeps `six_ability_suite_end_to_end_completed=false`.

## Observability, coordinate frames and units

Target visibility is computed only from the evaluation sidecar. For each time t we distinguish:

1. `visible`: visible now.
2. `seen_hidden`: seen at least once between the episode start and t, but not visible now.
3. `never_seen`: never seen up to t.

`causal_observability_masks(visibility_history)` accumulates only the prefix from the actual
episode reset. OR-ing the whole episode's visibility first is forbidden because it marks future
targets as seen in the past. Separate episodes are never concatenated before accumulating.
If official offline data lacks visibility, an oracle validated in a separate evaluator must be
built; the primary is never silently replaced by an "all targets" metric.

Target locations are read in the current egocentric frame. Self-localization and allocentric
maps use the episode's start position and heading as origin and axes. The camera coordinate
convention and the meter/maze-cell units are recorded in the manifest. Results obtained by
optimizing a rotation, translation or scale against test labels are never confused with direct
coordinate predictions.

`target_location_rmse` is `sqrt(mean(sum((prediction-target)^2, coordinate)))`, which differs
from a per-coordinate RMSE. The number of target observations in each observability group is
reported alongside, and an empty group's error is `null`, not 0. After computing the RMSE,
uncertainty is aggregated by episode and layout. Target or frame counts are not the independent
experimental N.

The wall map is evaluated only over cells actually observable up to t (`known_mask`).
Reconstruction of cells never seen yet does not enter the memory score. With wall = 1 and
free = 0, the balanced IoU is the macro average of wall IoU and free IoU. A class absent from both
prediction and truth is undefined and excluded. If `known_mask` is empty the score is `null`.
This keeps a solution that only gets the many free cells right from dominating overall accuracy.

Revisit segments must share the same training seed, layout, episode and target. A `match_bin`
for the initial oracle distance and target visibility is fixed before evaluation, and only the
unique first-visit record and the revisit records in the same bin are linked. Segment efficiency
is `success * oracle_distance / max(path_length, oracle_distance)`. Segments already at the goal
(distance 0) are excluded. Several revisits sharing one first visit are not independent samples.
The number of unmatched segments and the coverage must always be reported. `segment_id` is a
chronological nonnegative integer; revisits earlier than a successful first visit, or treating a
failed first visit as a real visit, are rejected. This metric is a matched association and does
not by itself prove causal use of memory, so hidden-state reset and history-truncation interventions
on the same policy are evaluated together.

## Splits and leakage prevention

The default RNG namespaces are below. The workers of one training seed may use
`seed * 1000 + worker`, which must stay below 1,000,000.

| Use | Seed range |
| --- | --- |
| RL train | 0–999,999 |
| RL validation | 1,000,000–1,999,999 |
| Policy final test | 2,000,000–2,999,999 |
| Probe train | 3,000,000–3,999,999 |
| Probe validation | 4,000,000–4,999,999 |
| Probe final test | 5,000,000–5,999,999 |

Different seeds do not guarantee different geometry. The canonical fingerprint of the actual maze
geometry is recorded and the intersection between splits is checked; which equivalences (rotation,
translation, ...) were removed must be recorded. Episode IDs are identified together with their layout.
A random frame split, or moving other trajectories of the same maze into the final test, is not
called scene or layout generalization.

`canonical_layout_fingerprint(grid)` treats the eight D4 symmetries of a binary wall/free grid,
including 90-degree rotations and reflections, as identical. Target placement, textures and RNG seeds
do not enter the hash. Translation and padding differences are not removed, so the grid export
convention must be fixed. Unless the layout inventories of both training and evaluation have been
collected, geometry disjointness across all RL and probe splits is not claimed.

`ProbeSplit` requires layout and episode provenance for every sample.
`fit_ridge_probe(train, validation)` rejects layout overlap and computes the mean and standard
deviation from train features only. The intercept is not penalized, and alpha is chosen among the
given candidates by validation MSE only. `evaluate_ridge_probe(model, test)` fails if train or
validation layouts appear in the final test. The input IDs must be real geometry fingerprints;
passing this check by renaming arbitrary IDs does not prove real geometry separation.

RL checkpoints are chosen only by validation return on the training task. We never pick the best
checkpoint separately for each spatial test metric, nor choose ridge alpha on the final test. Every
method gets the same probe label budget, sampling plan and capacity. The linear probe is primary; a
fixed small nonlinear probe is a separate auxiliary analysis.

## Information access versus behavioral ability

- **Diagnostic:** a supervised readout is trained on frozen features/state.
  That something can be decoded does not mean the policy actually uses it.
- **Readout transfer:** a new controller/head is trained on the frozen representation.
  It involves extra task supervision, so it is not functional zero-shot.
- **Functional zero-shot:** the actor is frozen too, and behavior is evaluated without any
  weight update on new mazes, long horizons and target revisits of the same task.

We use both common evaluation trajectories, where every method sees the same RGB/action history,
and real task episodes generated by each method's own policy. The former separates visited data
from update-rule effects; the latter measures behavioral outcomes including exploration. Common
histories are unrolled causally from the actual reset, never reusing another policy's cached hidden
state, and a probe sample's prefix is never zero-reset at an arbitrary length.

The default controls are untrained features, current-frame-only, action-only dead reckoning, a
train-mean predictor, hidden-state reset and history truncation. They check whether map or
localization information is explained by RGB appearance, an action clock or a trivial mean.

## Ablations and budgets

| Condition | Information used in training | Research interpretation |
| --- | --- | --- |
| AC+aux | Same RGB/action/reward | H1 comparison baseline |
| Original Disco | Same RGB/action/reward | Base result of the frozen discovered rule |
| Disco + corrected reward reuse | The same past trajectories and rewards | Replay / sample-efficiency effect |
| Positive-advantage self-imitation | Past successful chunks and the same reward | Objective change; the sample-based counterpart of distillation |
| Outcome/feedback-memory self-distillation | Signal the teacher builds from the agent's own rewards and RGB | Separate objective; per-run evidence gate |
| Shuffled-feedback control | Same compute and masks, feedback pairing destroyed | Control for feedback information |
| Privileged guided teacher | Pose/map/expert/new feedback | Separate guided study; no reward-only emergence claim |

An on-policy primary update and reuse of stale rollouts are different things. Replay must track
behavior-policy probabilities, age, importance correction and memory burn-in. A stored actor state
is never used as if it were the recurrent state of the current parameters. Reusing rewards alone
is not called an implementation of SDPO teacher-feedback distillation.

CNN and memory capacity, environment seeds, initialization and reward are matched. Extra replay or
teacher computation costs compute, so two panels are kept apart: (1) a sample budget matching fresh
transitions and (2) a compute budget matching total learner/teacher compute. Fresh and replayed
transitions, learner updates, teacher compute, wall time and peak memory are all reported. Equal
environment steps alone do not control for extra compute.

## Uncertainty and multiple comparisons

The current five training seeds are a **pilot/descriptive plan**. The confirmatory seed count and
seed IDs are not decided yet: they are `null` in the manifest and the design lock is `false`. After
the pilot, a power plan that accounts for the paired effects and variances, a scientifically meaningful
effect size and the multiple comparisons over the three primaries is locked together with an independent
confirmatory seed and layout bank. We never keep using the same seeds and test bank after seeing pilot
results and merely rename them confirmatory.

If the test candidate is an exact two-sided sign-flip test on training-seed paired differences, the
smallest possible p-value is `2 / 2^N` even in non-degenerate cases. **At N=5 it is 0.0625, so 0.05
significance is unattainable even before Holm.** At N=8 the minimum is 0.0078125, so 8 or more seeds
can be considered in power planning, but this concerns only the test's resolution and does not guarantee
sufficient power. Ties or zero differences can worsen the resolution further. The test assumes
sign-exchangeability of the paired differences under the null and is not claimed to be unconditionally
distribution-free.

Results of each method are paired by seed and layout. Targets/frames within an episode are aggregated
first, then averaged episode → layout → training seed. Bootstraps over seeds sharing the same evaluation
layouts preserve the layout pairing.

`aggregate_seed_clusters` requires episode-level scalar records and rejects duplicate records for one
episode. It returns the training-seed SD and a seed/layout hierarchical bootstrap CI. With one seed, the
across-training SD and CI are `null`. Intervals from few seeds are descriptive, and many frames are never
a reason to claim narrow certainty. This bootstrap CI describes seed/layout sampling variability; its tails
are never converted directly into significance, and frames are never permutation units. The p-value test
for method comparisons is specified separately in advance, with per-training-seed paired differences after
aggregating shared layouts as input. The current Holm helper only adjusts valid p-values; it does not
implement the sign-flip test or increase the power of five seeds.

After the confirmatory design is locked, **the three primary endpoints for H1's single contrast, Disco
versus AC+aux, form one family** with a Holm correction at 0.05. Alpha is not restarted at 0.05 for each
ability family. `holm_primary_tests` fails if given p-values other than these three endpoints. The test
that produces the p-values must itself follow the seed/layout pairing, and bootstrap CIs are never turned
into arbitrary p-values. Additional method contrasts, secondary assays and OOD axes cannot claim to have
passed this H1 family's correction; without a separate advance plan they are exploratory/descriptive.
H2's reuse-versus-distillation and H3's feedback comparisons form separate families while the teacher pilot
is still gated, and their endpoints and contrast scope must be re-registered before execution. OOD axes do
not mix size, horizon and appearance changes at once either.

## Execution gates and current implementation scope

1. **G0:** observation whitelist, numeric metrics, split/probe leakage, manifest tests.
2. **G1:** verification of real official benchmark reset/step and the recurrent learner connection.
3. **G2:** above-random task learning and history dependence in the pilot.
4. **G3:** a pilot-informed power plan and a lock of the independent confirmatory seed count, endpoint family and layout bank.
5. **G4:** H2/H3 only after teacher evidence passes; privileged variants are reported separately.

The foundations for G0 and G1 are in place: the visibility/known-map oracle, revisit telemetry and probe
collection are connected to the real benchmark. Still, no result can be claimed before the primary assays
run on research-scale checkpoints. The protocol manifest only validates the consistency of declarations;
it is not evidence of completed training or inference or of significant results.

Executable configs currently have the purpose `development` or `pilot`, and the rule-loss recipe is `fresh`
or `recent_replay`. Self-imitation and the feedback teacher are enabled by separate fields. Teacher
distillation is applied only after the per-run evidence gate passes (G4a: information gain over the shuffled
control; G4b: acting validation of the causal teacher). Confirmatory runs are disabled.
`runtime_end_to_end_verified` is never set to true without separate verification evidence.
The previous study is linked only through the [separate legacy archive note](LEGACY_ARCHIVE.md), and the
archive's results are never used as experimental results of this new protocol.
