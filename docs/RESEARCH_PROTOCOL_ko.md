# 제한된 reward에서 공간 능력의 창발을 측정하는 연구 프로토콜

[English](RESEARCH_PROTOCOL.md) | **한국어**

## 연구 질문과 주장 범위

한 개의 RGB navigation task를 제한된 성공 reward로 학습한 뒤, 동결한 visual
feature와 recurrent state에 어떤 공간 정보가 생기는지 측정한다. 모든 능력을
동시에 가르치는 multi-task supervised learner를 만드는 연구가 아니다.

세 가설을 순서대로 검증한다.

- **H1 — update rule:** 동일 CNN·memory·초기화·task에서 Disco가
  actor-critic + next-policy auxiliary보다 공간 표현을 개선하는가?
- **H2 — 경험 재활용:** 과거 trajectory의 기존 reward를 재사용한 효과에 더해,
  trajectory에서 만든 teacher의 distillation loss가 추가 이득을 주는가?
- **H3 — feedback의 정보:** 실제 teacher가 작동할 때, student가 가진
  RGB/action/reward에서 만든 feedback과 새로운 privileged 정보를 제공한
  feedback의 효과는 어떻게 다른가?

H1의 기본 learner는 depth, pose, 좌표, 지도, target vector, shortest path,
expert action을 입력이나 loss target으로 받지 않는다. task의 성공 여부를 채점하는
환경 내부 GT는 허용되지만, 이를 매 step의 거리 shaping이나 teacher 답안으로
전달하면 다른 실험이 된다. task goal의 색·이름은 task supervision이다.

**SDPO라는 이름을 붙인다고 teacher가 검증되는 것은 아니다.**
[SDPO 원 논문](https://arxiv.org/abs/2601.20802)을 공간 task에 이식한 self-teacher는
구현되어 있다([설계](SDPO_SPATIAL_RESEARCH_PLAN_ko.md)). 하지만 manifest는 `validated=false`로
선언한다. 증류는 run마다 held-out 증거 gate를 통과한 뒤에만 적용된다. 다음을 모두 확인하기 전에는
end-to-end SDPO 결과라고 부르지 않는다: teacher input provenance, shuffled 통제보다 나은
정보량, 실제 행동 우위, distillation gradient, replay·self-imitation control, compute control.

## 기존 benchmark와 자체 진단의 경계

공간 메모리 자체를 연구할 때는 [Memory Maze](https://github.com/jurgisp/memory-maze)가
직접적인 출발점이다. 표준 task는 같은 episode 동안 고정된 maze와 target 위치에서
색으로 지정한 target에 닿을 때마다 +1을 받고 다음 target을 찾는다. 이는 sparse
task-success reward지만 **episode terminal-only reward와 동일하지 않다**.
이번 주조건은 upstream의 target 접촉 +1/나머지 0을 그대로 유지하며 **step cost를
추가하지 않는다**. 성공 종료 규칙도 변경하지 않는다. 다른 reward를 연구하면 별도
protocol variant로 선언하며 이번 주조건의 결과와 섞지 않는다.

엄격한 terminal success + step cost를 유지하려면
[RoboTHOR ObjectNav](https://github.com/allenai/robothor-challenge)의 기존
RGB-only task를 별도 study로 사용할 수 있다. task definition, build, episode,
success/SPL evaluator를 고정한다. [공식 AllenAct 구현](https://allenact.org/projects/objectnav_baselines/)
의 RGB baseline도 memory를 사용하지만, pretrained feature나 geodesic shaping을
그대로 가져와 geometry-free random-init 실험과 섞지 않는다.

Memory Maze의 기존 online return 및 wall/object-location probing 개념,
RoboTHOR의 success/SPL은 공식 평가에 기반한다. 아래의 관측가능성 분리,
balanced map IoU, matched revisit gain, relative pose, loop closure,
reachability readout은 **이 연구가 정의한 추가 diagnostic**이다.
모든 지표를 공식 benchmark 점수처럼 부르지 않는다.

## 능력군과 고정 primary endpoint

| 능력군 | 평가 대상 | 상태·평가 성격 |
| --- | --- | --- |
| Local geometry | 렌더 depth, ±30° free-space, 전진 가능성 | 추가 diagnostic; image-feature readout 구현 |
| Self-localization | 상대 이동·heading, episode-origin 위치, drift | 추가 diagnostic; action-history 통제 포함 구현 |
| Spatial memory | 과거 봤으나 현재 안 보이는 target 위치 | **Primary 1:** `seen_hidden_target_location_rmse` (segmentation visibility) |
| Place recognition / loop closure | 긴 시간차의 같은 장소 retrieval | 추가 diagnostic; 학습 없는 recall@1과 chance rate |
| Map topology | 관측한 wall map, 미관측 cell 분리 | **Primary 2:** `known_wall_balanced_iou` (depth 역투영 known mask); reachability는 후속 |
| Navigation | 원 task 행동과 효율, target 재방문, memory 개입 | **Primary 3:** `matched_revisit_efficiency_gain` |

depth와 free-space, map과 reachability 등의 상관된 지표를 각각 독립적인
“새 능력”으로 세지 않는다. OOD는 일곱 번째 능력이 아니라 각 능력군의 일반화 축이다.
여섯 능력군의 collector, suite, navigation, seed report는 구현되어 실제 benchmark에서
smoke 규모로 실행됐다. 그러나 연구 규모 학습 checkpoint에서 final report까지 실행한 것은
아니다. manifest는 `six_ability_suite_end_to_end_completed=false`로 남는다.

## 관측가능성, 좌표계와 단위

target visibility는 evaluation sidecar에서만 계산한다. 시점 t에 대해 다음을 구분한다.

1. `visible`: 현재 보임.
2. `seen_hidden`: episode 시작부터 t까지 한 번 이상 봤으나 현재는 안 보임.
3. `never_seen`: t까지 한 번도 보지 못함.

`causal_observability_masks(visibility_history)`는 실제 episode reset부터의
prefix 누적만 사용한다. episode 전체의 visibility를 먼저 OR하면 미래 target을
과거에 본 것으로 표시하므로 금지한다. 서로 다른 episode를 연결해서 누적하지 않는다.
official offline 데이터에 visibility가 없으면 별도 evaluator에서 검증한 oracle를
만들어야 하며, 해당 primary를 임의로 “모든 target” 지표로 대체하지 않는다.

target 위치는 current egocentric frame의 좌표로 읽는다. self-localization이나
allocentric map은 episode 시작 위치·방향을 원점과 기준축으로 사용한다.
카메라 좌표 convention과 meter/maze-cell 단위를 manifest에 남긴다.
test label로 회전·이동·scale을 최적화해 맞춘 결과를 직접 좌표 예측과 혼동하지 않는다.

`target_location_rmse`는 `sqrt(mean(sum((prediction-target)^2, coordinate)))`이다.
이는 coordinate별 RMSE와 다르다. 각 관측군의 target observation 수를 같이 보고하고,
빈 군의 error는 0이 아닌 `null`로 남긴다. RMSE를 구한 뒤 uncertainty는 episode와
layout 단위로 집계한다. target/frame 개수는 독립적인 실험 N이 아니다.

wall map은 t까지 실제 관측 가능한 cell만 `known_mask`로 평가한다. 아직 한 번도
보지 못한 cell의 reconstruction은 memory 점수에 넣지 않는다. target wall을 1,
free를 0으로 두고 wall IoU와 free IoU의 macro 평균을 balanced IoU로 정의한다.
prediction과 truth 양쪽에 모두 없는 class는 undefined로 제외한다. `known_mask`가
비면 점수는 `null`이다. 많은 free cell만 맞히는 해법이 전체 accuracy를 지배하지 않게 한다.

revisit segment는 같은 training seed, layout, episode, target을 공유해야 한다.
초기 oracle 거리·target 가시성에 대한 `match_bin`을 평가 전에 고정하고, 같은 bin의
유일한 first-visit record와 revisit record만 연결한다. segment efficiency는
`success * oracle_distance / max(path_length, oracle_distance)`다.
거리 0의 이미 도착한 segment는 제외한다. 같은 first visit를 공유한 여러 revisit는
독립 표본이 아니다. unmatched segment 수와 coverage를 반드시 보고한다.
`segment_id`는 시간 순서의 nonnegative integer이며, 성공한 first visit보다
이른 record나 실패한 first visit을 실제 방문으로 취급한 revisit는 거부한다.
이 지표는 matched association이며 memory의 인과적 사용을 단독 증명하지 않는다.
동일 policy의 hidden-state reset / history truncation 개입을 함께 평가한다.

## split과 leakage 방지

기본 RNG namespace는 다음과 같다. 한 training seed의 worker는
`seed * 1000 + worker`를 사용할 수 있지만 1,000,000 미만이어야 한다.

| 용도 | seed 범위 |
| --- | --- |
| RL train | 0–999,999 |
| RL validation | 1,000,000–1,999,999 |
| Policy final test | 2,000,000–2,999,999 |
| Probe train | 3,000,000–3,999,999 |
| Probe validation | 4,000,000–4,999,999 |
| Probe final test | 5,000,000–5,999,999 |

서로 다른 seed가 서로 다른 geometry를 보장하지는 않는다. 실제 maze geometry의
canonical fingerprint를 기록하고 split 간 교집합을 검사한다. 회전·이동 등 어떤
동치를 제거했는지 기록해야 한다. episode ID는 layout와 합쳐 식별한다.
frame random split이나 같은 maze의 다른 trajectory를 final test로 옮기는 것을
scene/layout generalization이라고 부르지 않는다.

`canonical_layout_fingerprint(grid)`는 binary wall/free grid의 90도 회전과 reflection을
포함한 D4 대칭 8개를 동일하게 처리한다. target 배치·texture·RNG seed는 hash에
들어가지 않는다. translation이나 padding 차이는 제거하지 않으므로 grid export
convention을 고정해야 한다. 학습과 평가의 layout inventory가 모두 수집되지 않았다면
RL/probe 전체의 geometry disjointness를 검증했다고 주장하지 않는다.

`ProbeSplit`은 모든 sample에 layout·episode provenance를 요구한다.
`fit_ridge_probe(train, validation)`는 layout overlap을 거부하고 train feature로만
평균·표준편차를 구한다. intercept에는 penalty를 주지 않으며, alpha는 지정한
candidate 중 validation MSE로만 고른다. `evaluate_ridge_probe(model, test)`는
train 또는 validation layout이 final test에 섞이면 실패한다.
입력 ID는 실제 geometry fingerprint여야 한다. 임의의 이름을 바꾸는 것으로
이 검사를 통과시켜도 진짜 geometry 분리가 증명되는 것은 아니다.

RL checkpoint는 training-task validation return으로만 고른다. 여러 spatial
test metric에서 각각 최고 checkpoint를 뽑거나, final test로 ridge alpha를 고르지 않는다.
모든 method에 같은 probe label budget·sampling plan·capacity를 적용한다.
linear probe가 primary이며 고정 소형 nonlinear probe는 보조 분석으로 구분한다.

## 정보 접근성과 행동 능력의 구분

- **Diagnostic:** 동결한 feature/state에 supervised readout을 학습한다.
  decode된다고 policy가 실제로 그 정보를 쓰는 것은 아니다.
- **Readout transfer:** 동결 표현에 새 controller/head를 학습한다.
  추가 task supervision이 있으므로 functional zero-shot이 아니다.
- **Functional zero-shot:** actor까지 동결하고 추가 weight update 없이
  새로운 maze·long horizon·같은 task의 target 재방문에서 행동을 평가한다.

모든 method가 같은 RGB/action history를 보는 공통 evaluation trajectory와,
각자 policy로 만든 실제 task episode를 함께 사용한다. 전자는 visited data와
update-rule 효과를 분리하고, 후자는 exploration을 포함한 행동 결과를 측정한다.
공통 history는 실제 reset부터 causal하게 unroll하며 다른 policy의 cached hidden
state를 재사용하지 않는다. probe sample의 prefix를 임의 길이에서 zero reset하지 않는다.

기본 control은 untrained feature, current-frame-only, action-only dead reckoning,
train-mean predictor, hidden-state reset, history truncation이다. 지도·localization의
정보가 RGB appearance, action clock 또는 낮은 난도의 평균값으로 설명되는지 확인한다.

## ablation과 예산

| 조건 | 학습에 사용한 정보 | 연구 해석 |
| --- | --- | --- |
| AC+aux | 같은 RGB/action/reward | H1 비교 기준 |
| Original Disco | 같은 RGB/action/reward | frozen discovered rule의 기본 결과 |
| Disco + corrected reward reuse | 과거의 같은 trajectory와 reward | replay/sample-efficiency 효과 |
| Positive-advantage self-imitation | 과거 성공 chunk와 같은 reward | 목적함수 변경; 증류의 sample 기반 대응물 |
| Outcome/feedback-memory self-distillation | teacher가 자기 trajectory의 reward·RGB에서 만든 신호 | 별도 objective; run별 증거 gate |
| Shuffled-feedback control | 같은 연산과 mask, feedback 쌍 파괴 | 피드백 정보량 통제 |
| Privileged guided teacher | pose/map/expert/new feedback | 별도의 guided study; reward-only emergence 주장 금지 |

on-policy primary update와 stale rollout 재사용은 별개다. replay에선 behavior policy
확률·age·importance correction·memory burn-in을 추적해야 한다. stored actor state를
현재 parameter의 recurrent state인 것처럼 사용하지 않는다. reward 재활용만으로
SDPO teacher-feedback distillation을 구현했다고 부르지 않는다.

CNN·memory capacity, 환경 seed, 초기화, reward는 맞춘다. 추가 replay/teacher는
연산을 쓰므로 두 panel을 구분한다: (1) fresh transition 수를 맞춘 sample budget,
(2) total learner/teacher compute를 맞춘 compute budget. fresh·replayed transition,
learner update, teacher compute, wall time, peak memory를 모두 보고한다.
환경 step만 같다고 추가 연산이 통제되는 것은 아니다.

## uncertainty와 다중 비교

현재 training seed 5개는 **pilot/descriptive 계획**이다. confirmatory seed 수와
seed ID는 아직 정하지 않았으며 manifest에서 `null`, design lock은 `false`다.
pilot의 paired effect·variance, 과학적으로 의미 있는 effect size, primary 세 개의
다중 비교를 고려한 power 계획을 만든 뒤 독립된 confirmatory seed/layout bank와
함께 lock한다. pilot 결과를 본 뒤 같은 seed와 test bank를 계속 사용하며
confirmatory라고 이름만 바꾸지 않는다.

검정 후보를 training-seed paired difference에 대한 exact two-sided sign-flip으로
정하면, 가능한 최소 p-value는 비퇴화한 경우에도 `2 / 2^N`이다. **N=5에서는
0.0625여서 Holm 이전에도 0.05 유의성을 얻을 수 없다.** N=8에서는 최소
0.0078125이므로 8개 이상을 power planning의 후보로 검토할 수 있지만, 이것은
검정 해상도에 관한 조건일 뿐 충분한 power를 보장하지 않는다. ties/zero difference는
해상도를 더 나쁘게 만들 수 있다. 이 검정은 null 아래 paired difference의
sign-exchangeability를 가정하며 무조건 distribution-free라고 주장하지 않는다.

각 method의 동일 seed·layout 결과를 pairing한다. episode 안의 target/frame를 먼저
집계하고, episode→layout→training seed 순서로 평균한다. 같은 평가 layout를 공유하는
seed의 bootstrap에서는 layout pairing을 보존한다.

`aggregate_seed_clusters`는 episode scalar record를 요구하며 같은 episode의 중복
record를 거부한다. training-seed SD와 seed/layout hierarchical bootstrap CI를
반환한다. seed 1개이면 across-training SD/CI는 `null`이다. 작은 seed 수의 interval은
descriptive이며 수많은 frame가 있다는 이유로 좁은 확신을 주장하지 않는다.
이 bootstrap CI는 seed/layout sampling variability의 기술 통계다. CI의 tail을
그대로 유의확률로 바꾸거나, frame를 permutation unit으로 사용하지 않는다.
method comparison의 p-value 검정은 공통 layout를 집계한 training-seed별 paired
difference를 입력으로 별도 사전 지정한다. 현재 코드의 Holm helper는 유효한
p-value를 조정할 뿐 sign-flip 검정을 대신 구현하거나 seed 5개의 power를 늘리지 않는다.

confirmatory design을 lock한 뒤 **Disco 대 AC+aux라는 H1의 한 contrast에 대해
세 primary endpoint 전체를 한 family**로 묶어 0.05 Holm correction을 사용한다.
각 ability family별로 alpha를 0.05로 다시 시작하지 않는다.
`holm_primary_tests`는 이 세 endpoint 이외의 p-value를 넣으면 실패한다.
p-value를 만드는 검정 자체도 seed/layout pairing을 따라야 하며 bootstrap CI를
임의의 p-value로 변환하지 않는다. 추가 method contrast, secondary assay, OOD 축은
이 H1 family의 보정을 통과했다고 주장할 수 없다. 별도 사전 계획이 없으면
exploratory/descriptive다. H2의 reuse 대 distillation, H3의 feedback 비교는 teacher
pilot가 아직 gated인 별도 family이며, 실행 전에 endpoint·contrast 범위를 다시
등록해야 한다. OOD 축도 크기·horizon·appearance 변화를 한 번에 섞지 않는다.

## 실행 gate와 현재 구현 범위

1. **G0:** observation whitelist, numeric metric, split/probe leakage, manifest tests.
2. **G1:** 실제 official benchmark reset/step와 recurrent learner 연결 검증.
3. **G2:** pilot에서 random을 넘는 task learning과 history 의존성 확인.
4. **G3:** pilot-informed power 계획과 독립 confirmatory seed 수·endpoint family·layout bank lock.
5. **G4:** teacher 증거를 통과한 뒤 H2/H3; privileged variant는 따로 보고.

G0와 G1의 기반은 갖춰졌다. Visibility/known-map oracle, 재방문 telemetry, probe
collection이 실제 benchmark에 연결되어 있다. 다만 연구 규모의 checkpoint에서 primary assay를
실행하기 전에는 결과를 주장할 수 없다. protocol manifest는 선언의 일관성을 검증하는 파일일 뿐이다.
완료된 학습·추론이나 유의한 결과의 증거가 아니다.

현재 executable config의 용도는 `development` 또는 `pilot`이다. Rule-loss recipe는 `fresh` 또는
`recent_replay`다. Self-imitation과 feedback teacher는 별도 field로 켠다. Teacher 증류는
run별 증거 gate(G4a: shuffled 통제 대비 information gain, G4b: 인과적 teacher의 행동 검증)를
통과할 때만 적용된다. Confirmatory 실행은 비활성이다.
`runtime_end_to_end_verified`는 별도 검증 증거 없이 true로 바꾸지 않는다.
이전 연구는 [독립 legacy archive 안내](LEGACY_ARCHIVE_ko.md)에만 연결하며,
archive의 결과를 이 새 protocol의 실험 결과로 사용하지 않는다.
