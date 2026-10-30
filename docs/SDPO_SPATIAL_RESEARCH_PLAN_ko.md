# 제한된 reward에서의 공간 능력 형성과 SDPO형 사후 피드백 증류: 연구 설계

[English](SDPO_SPATIAL_RESEARCH_PLAN.md) | **한국어**

## 연구 질문

**같은 제한된 reward와 상호작용 예산에서 Disco가 어떤 공간 능력을 형성하는가? 경험 재사용과 사후 피드백은 그 능력을 어떻게 바꾸는가?**

TripoSplat 기반 물체 실험은 주 실험에서 제외했다([legacy archive](LEGACY_ARCHIVE_ko.md)). 이 문서는 능력 평가 체계와 RL 학습 규칙의 개발 계획을 다룬다. 통계·split·leakage 규약은 [연구 프로토콜](RESEARCH_PROTOCOL_ko.md), 구현 세부는 [METHODS](METHODS_ko.md)를 따른다.

## 벤치마크

| 역할 | 환경 | 이유 |
| --- | --- | --- |
| 주 실험 | Memory Maze 9×9 RGB | 지정 물체에 닿을 때마다 +1, 다음 목표 지정. 한 episode 안에서 미로와 물체 위치가 고정되므로 기억·위치 추적·지도 형성·재방문을 함께 볼 수 있다. |
| 후속 검증 | RoboTHOR RGB ObjectNav | 실제적인 장면에서 terminal success와 SPL로 일반화를 확인한다. pretrained feature나 geodesic shaping과 섞지 않는다. |

Memory Maze의 reward는 episode당 한 번의 terminal reward가 아니다. 목표 도달마다 받는 한 종류의 보상이다. 주 조건은 upstream reward를 그대로 쓴다. step cost나 거리 shaping은 추가하지 않는다.

## 능력군과 측정 구현

평가 라벨은 evaluation split의 sidecar에서만 만든다. 학습자 입력에는 들어가지 않는다. Visibility와 known map은 actor 카메라를 같은 자세에서 다시 렌더링해 얻는다. Segmentation으로 target 픽셀을 세고, depth를 역투영해 관측된 cell을 구한다. 이렇게 다시 렌더링한 RGB는 관측과 픽셀 단위로 일치한다. 실제 episode에서 floor 픽셀의 99.8%가 free cell에, wall 픽셀의 99.7%가 wall cell에 떨어졌다. 보이는 target의 방위각은 모두 시야각 ±40°에 구 반경을 더한 범위 안에 있었다.

| 능력군 | 측정 | 구현 |
| --- | --- | --- |
| 자기 위치·이동 추정 | 시작 자세 기준 위치·heading, 1·8 step 상대 이동, 시간 구간별 drift | `suite.self_localization` |
| 공간 기억 | **보였다가 지금 안 보이는(seen-hidden) target의 egocentric 위치 RMSE** (primary 1) | segmentation visibility의 인과적 누적 mask |
| 장소 재인식 | 50 step 이상 떨어진 같은 장소·방향의 nearest-neighbor recall@1과 chance rate | 학습 없는 cosine retrieval |
| 지도·공간 관계 | **지금까지 관측한 cell의 wall/free balanced IoU** (primary 2), 미관측 cell 점수 분리 | 시작 cell·시작 heading의 grid축 기준 canvas |
| 국소 기하·행동 가능성 | 렌더 depth 8×8 RMSE, ±30° free-space 거리, 전진 가능성 AP | 단일 frame image feature |
| 실제 공간 행동 | return, **매칭된 재방문 효율 향상** (primary 3), memory reset과 history truncation 개입 | 동결 정책의 test episode |

새 layout, 더 긴 horizon, 큰 미로, 시각 변화는 별도 능력이 아니다. 각 능력의 일반화 축이다.

### 증거의 세 수준

1. **정보가 있다:** 동결 belief에서 작은 선형 probe로 읽힌다. 통제군은 image feature만 쓰는 frame-only, RGB 없는 action history, 같은 초기화의 untrained `init.pkl`, train-mean이다.
2. **전이 가능하다:** 동결 표현 위에 새 controller를 학습해 활용한다. 추가 supervision이 있으므로 zero-shot이 아니다.
3. **실제 능력이다:** actor까지 동결한 채 행동이 좋아진다. 재방문 효율이 오르고, memory reset을 하면 성능이 떨어진다.

지도 probe 하나가 좋아졌다고 planning 능력이 생겼다고 결론 내리지 않는다.

## SDPO에서 가져오는 것

[SDPO](https://arxiv.org/abs/2601.20802)의 핵심은 과거 성공 행동을 반복 학습하는 데 있지 않다. 피드백을 본 정책의 분포 자체를 학습 신호로 쓴다는 점이다. 원 논문에서 확인한 요소와 이 연구의 대응은 다음과 같다.

| SDPO (LLM) | 이 연구 (recurrent RGB 정책) |
| --- | --- |
| 현재 모델이 feedback을 context에 넣고 teacher가 된다 | 학생 GRU belief(stop-gradient)에 feedback head를 붙인 self-teacher |
| 손실은 KL(student ‖ stopgrad(teacher)), token마다 | KL(student ‖ stopgrad(teacher)), action step마다, 6개 action 전체 분포 |
| EMA teacher(α=0.01)로 정규화한다. 정규화가 없으면 발산 | teacher head의 EMA(`teacher_ema_rate=0.01`)로 target을 만든다 |
| scalar reward에서는 같은 문제의 다른 성공 rollout을 feedback으로 쓴다(자기 자신 제외) | 같은 episode·같은 target 색의 이전 성공 frame을 쓴다(`success_memory`, 인과적, 자기 자신 제외) |
| feedback은 시도에 대한 사후 정보다 | 시도 후 H step 안에 reward가 왔는지(`outcome`, 자기 시도에 대한 hindsight) |
| 실험은 strict on-policy | 각 최신 chunk를 outcome window가 다 관측된 직후 한 번만 쓴다(`MaturedChunks`, 나이를 기록) |

Memory Maze는 reset마다 새 미로를 만든다. 따라서 "같은 task instance"는 한 episode와 한 target 색이다. 이것이 logical instance와 attempt ID 문제에 대한 이번 설계다.

### Self-teacher를 학습해야 하는 이유와 구조

LLM teacher는 이미 피드백을 해석할 수 있다. 처음부터 학습하는 CNN/GRU에는 그런 능력이 없다. 그래서 피드백 경로를 명시적으로 학습한다.

```text
teacher_logits = base_logits + Δ_ψ(stopgrad(belief_t), feedback_t)
학습:   behavior logits를 base로, 실제 행동을 behavior cloning (관측된 outcome으로 조건화)
target: stopgrad(student logits) + Δ_EMA(belief_t, query feedback)
```

Bayes 규칙에 따라 Δ의 최적값은 log P(feedback | history, action)이다(step별 상수는 무시). 이 형태에서 세 가지 결과가 나온다.

1. **결과로 조건화하지 않으면 teacher는 아무것도 배우지 못한다.** On-policy data에서 행동은 history 밖의 인과적 정보와 조건부 독립이다. 과거 성공 frame만 넣은 teacher는 Δ=0으로 수렴한다. 그래서 모든 teacher가 `outcome` 채널을 가진다. `success_memory`는 outcome과 결합될 때만 의미가 있다.
2. **success를 query하면 soft policy improvement가 된다.** Target이 π(a|h)·P(success | h, a)에 비례한다. 단위 테스트에서 이 posterior를 오차 0.04 안에서 복원했다(`tests/test_teacher.py`).
3. **self-imitation과 기댓값이 같다.** 성공 step에서 이 target의 기댓값은 성공 조건부 모방(SIL)의 기댓값과 같다. 즉 SDPO형 증류는 SIL을 Rao–Blackwell화한 형태다. 따라서 **H2의 예측**은 이렇다. FD가 SIL보다 낫다면 그 이유는 분산 감소와 teacher의 일반화여야 한다. 추가 reward 정보로는 설명할 수 없다. 둘이 같다면 이 이론과 일치한다.

Query는 두 가지다. `actual_success`는 실제로 성공한 step에만 증류한다. Hindsight이므로 teacher가 직접 행동할 수 없다. `counterfactual_success`는 모든 step에서 "곧 성공한다"를 query한다. Query가 상수이고 memory가 인과적이므로 이 teacher는 **직접 행동할 수 있다**. 그래서 student와 같은 validation episode에서 직접 비교할 수 있다(`evaluate --teacher`).

### 과거 rollout이 쓰이는 두 위치

| 방식 | 과거 rollout의 역할 | 구현 |
| --- | --- | --- |
| Feedback memory | 최신 attempt를 학습할 때 teacher의 참고자료 | `success_memory` 채널. 같은 episode의 이전 성공 frame |
| Sequence replay | 과거 rollout 자체에서 다시 loss와 gradient를 계산 | `recipe=recent_replay`. 원 behavior logits와 현재 parameter로 prefix를 재구성 |

전체 손실은 다음과 같다.

```text
L = L_Disco,fresh [+ α·L_Disco,replay] + λ_SIL·L_SIL + λ_T·L_teacher-BC + λ_SD·g·KL(student ‖ sg(teacher_EMA))
```

g는 아래 증거 gate다. 이 손실은 공개 Disco103에 덧붙인 확장으로 보고한다. Meta-RNN, advantage EMA, target EMA 같은 rule 내부 상태는 fresh 데이터로만 갱신한다.

## Teacher가 유용한지 검증하는 방법

Teacher가 학습하기 전의 chunk에서 매 update마다 진단을 계산한다. 즉 held-out 진단이다. Gate는 이 진단의 지수평균이 네 조건을 모두 만족할 때만 열린다.

| 진단 | 의미 | gate 조건 |
| --- | --- | --- |
| information gain (success step) | E[log teacher(a) − log μ(a)]. 피드백이 성공 행동을 설명하는 정도 | ≥ `teacher_min_information_gain` |
| shuffled information gain | 같은 step에 feedback을 섞어 넣은 값 | gain과의 차이(margin) ≥ 같은 임계값 |
| local improvement | E[(teacher(a)/μ(a) − 1)·G^H]. 1차 정책 개선 추정 | > 0 |
| copy rate | teacher가 실제 행동에 0.95 이상을 주는 비율. 행동 복사 감지 | ≤ `teacher_max_copy_rate` |

조건이 나빠지면 gate는 다시 닫힌다. 여닫은 이력은 `completed.json`에 기록된다. 추가로 인과적 teacher(success query와 memory)가 validation episode에서 직접 행동해 student와 비교한다(G4b). `teacher_gate=never`는 teacher를 학습만 하고 증류하지 않는 검증 조건이다. `always`는 gate를 강제로 여는 ablation이다.

Teacher 입력은 `own_reward_outcome_within_horizon`, `own_rgb_frame_of_previous_same_color_success`, `elapsed_steps`, `stopgrad_student_belief`뿐이다. protocol validator가 이를 강제한다. GT map·pose·depth·최단 경로를 주는 teacher는 추가 공간 감독이다. 그런 조건은 별도의 guided study로만 다룬다(A4, 미구현).

## 비교 조건과 해석 규칙

조건표는 [`experiments/conditions.json`](../experiments/conditions.json)에 있다. `scripts/expand_conditions.py`가 이를 seed별 config로 펼친다. 각 조건은 Disco와 AC+aux(같은 recurrent backbone, 같은 seed별 초기화)에 모두 적용한다.

| 조건 | 무엇이 바뀌는가 | 신호의 종류 |
| --- | --- | --- |
| fresh | 없음 | 기준 |
| replay | 과거 sequence에서 rule loss | 같은 reward 재사용 |
| self_imitation | 성공 chunk의 positive-advantage 모방 | 목적함수 변경, 같은 reward |
| fd_outcome | outcome teacher 증류 | 같은 reward로 만든 사후 피드백 경로 |
| fd_memory | outcome과 이전 성공 frame, success query | 피드백 memory |
| fd_outcome_shuffled | 같은 연산·mask, feedback 쌍을 파괴 | 피드백 정보량 통제 |
| fd_outcome_replay | 증류와 sequence replay 결합 | 두 축 결합 |
| teacher_only | teacher를 학습·검증만 함 | teacher 검증 |

"같은 reward를 더 잘 활용한 효과"와 "추가 학습 신호를 준 효과"를 구분해 표기한다. 관측 기반 보조 목적은 아직 추가하지 않았다. 예를 들어 action-conditioned prediction은 새 self-supervised objective이고, coverage bonus는 reward 변경이다. 이런 조건을 추가하면 별도 축으로 표시한다. 환경 interaction 수와 함께 `learner_network_steps`(learner가 수행한 관측 단위 network 평가의 근사 수)와 wall time을 보고한다. 이를 근거로 sample budget과 compute budget 두 panel로 비교한다.

## 개발 순서와 실행 gate

1. Memory Maze 통합. 실제 reset/step과 render telemetry 검증 — **완료**
2. Recurrent 학습 경로 정비. Block 단위 prefix 재구성, vectorized replay, `init.pkl` — **완료**
3. 공통 ability 평가. Telemetry collector, 여섯 능력군 suite, navigation, seed 집계 report — **구현과 smoke 검증 완료. 연구 규모 실행 전**
4. Replay와 self-imitation baseline — **구현, smoke 검증**
5. Self-teacher 검증. Held-out 진단, shuffled 통제, gate, teacher acting — **구현. 고정 explore 정책의 offline 사전 점검에서는 정보 이득이 없었다(아래). RL 학습 중 유용성은 미검증**
6. SDPO 계열 결합. Fresh with lookahead, replay 결합, memory 채널 — **구현, smoke 검증**

`P1_learning_gate`는 2026-10-28부터 GPU 서버(Vast.ai L40S)에서 12개 run(Disco·AC+aux × fresh·teacher_only × 3 seeds)을 각 5M step 돌렸다. **G2는 통과하지 못했다.** 모든 run의 정책 엔트로피가 처음 약 0.5~1M step 안에 ln 6 ≈ 1.792, 즉 균일 정책에 도달했다. 학습 보상도 균일 정책 수준(1000-step episode당 약 0.2)에 머물렀다. 한 run은 `forward_left` 쪽으로 치우친 동안 보상률이 잠시 두 배가 됐다가 다시 균일 분포로 돌아갔다. Test return(run당 8 episode)은 random과 구분되지 않았다. Teacher의 정보 이득은 셔플 대조군과 1e-9까지 같았다. 즉 Δ head가 0에 머물렀고 G4a도 통과하지 못했다. 학습 후 frozen probe 능력은 untrained init보다 나을 게 없었고, 일부는 더 나빴다(loop closure recall 0.38 → 0.31~0.33). 모두 pilot 증거이며 결과는 `runs/pilot_results`에 있다.

P2는 학습기가 실제로 학습할 때까지 미룬다. `P1d_diagnostics`는 P1이 왜 G2를 통과하지 못했는지를 세 가지로 확인한다.

* **Positive control:** `cue_recall`은 cue 색이 보상받는 행동을 알려 주는 합성 과제다. 같은 step에 알려 주는 `control_reactive`와 8 step 뒤에 알려 주는 `control_memory`가 있다. cue를 고정하면 지각이 전혀 필요 없는 단일 정답 bandit(`control_bandit`)이 된다. 두 학습기가 이 과제들을 통과해야 Memory Maze 결과를 해석한다. development 용도로만 쓸 수 있고, 연구 과제로는 쓰지 않는다.
* **Credit 진단**(`credit_window=8`): update마다 보상 직전 step과 그렇지 않은 step에서 택한 행동의 log-prob이 각각 얼마나 오르는지 비교한다. 규칙 자체의 신호도 기록한다. Disco는 target log-prob 이득, AC는 advantage다.
* **Entropy ablation:** AC+aux를 entropy cost 0.001과 0으로 돌려 P1의 0.01과 비교한다.
* **Disco103 learning rate:** Disco를 3e-3과 1e-2(업스트림 평가 colab의 값)로 돌려 P1의 3e-4와 비교한다. Memory Maze와 positive control 모두에서 비교한다.

단계 실행 전 로컬 CPU 점검은 작은 네트워크(channels 16/32, hidden 64)와 seed 1개로 했으므로 development 증거일 뿐이다.

* **Bandit:** AC+aux는 3e-4에서 약 100 update 만에 풀었다. Disco103은 3e-4에서 200 update 동안 오히려 정답에서 *멀어졌다*. 3e-3에서는 약 250 update 동안 균일 정책 정체(엔트로피 1.792)에 머문 뒤 풀었고, 1e-2에서는 약 300 update 만에 풀었다.
* **업스트림 harness:** 수정하지 않은 업스트림 Agent도 똑같이 행동했다. 3e-4 bandit에서는 오답 행동으로 붕괴했다. 업스트림 Catch에서는 3e-4로 400 update 동안 학습이 없었고, 1e-2로는 약 300 update 뒤부터 학습했다(`scripts/upstream_disco_check.py`). 즉 공개 규칙은 이 소프트웨어 스택에서도 작동하지만, P1과 동떨어진 learning rate 영역에서만 작동한다.
* **시각 cue 과제**(`control_reactive`, 300k step): 두 학습기 모두 풀지 못했다. AC+aux는 step당 보상 0.21~0.23에 그쳤다(chance 0.17, optimum 1). Disco는 3e-3과 1e-2에서 cue와 무관한 행동 하나로 수렴했고, encoder 특징의 cue 간 차이가 3.6에서 0.25로 붕괴했다. 초기 네트워크는 cue를 구분하므로(belief 차이 1.45), 정보는 처음부터 없던 것이 아니라 Disco 학습 중에 사라진 것이다.

이에 맞춰 positive control 예산을 늘렸다(`control_reactive` 2M step). 점수가 매겨지는 행동은 바로 다음 step에 보상을 받으므로 credit window는 1로 둔다.

P1의 붕괴가 그 안에 일어났으므로 step 수는 2M으로 하고, validation은 sampled로 한다. 그다음 `scripts/diagnose_stage.py`가 on-policy 학습 보상을 같은 test seed의 scripted 기준선(uniform, `forward_left`·`forward` 편향)과 비교한다. Test 평가도 32 episode 이상으로 늘리고, random 기준선과 같은 sampled 행동으로 맞춘다(`evaluate_stage.py --test-seeds 32 --policy sampled`).

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

## 지금까지 실행한 것 (연구 결과 아님)

요약은 [`reports/development/smoke_summary.json`](../reports/development/smoke_summary.json)에 있다. 원자료는 gitignore된 `runs/smoke`, `runs/teacher_validation`, `datasets/smoke`에 있다.

- **실제 Memory Maze smoke matrix.** 일곱 조건을 각각 2,048 step 학습했다(fresh, AC+aux, replay, self-imitation, outcome teacher, memory teacher with forced gate, shuffled 통제). 모두 유한한 loss로 끝났다. 학습 중 reward는 0회였다. 그래서 self-imitation은 reward chunk를 찾지 못했고, 증거 gate는 닫힌 채였다(증류 0회). Success 경로는 단위 테스트로만 검증되어 있다. `disco-sil`은 `disco-fresh`와 비트 단위로 같았다. Auxiliary batch가 만들어지지 않았기 때문이며, 공유 초기화와 결정성이 지켜진다는 확인이다.
- **Ability suite와 navigation.** Telemetry collection(학습 4, 검증 2, test 2 episode, 모두 다른 layout)에서 feature 추출, suite, navigation, report까지 실제 데이터로 실행됐다. Smoke checkpoint의 seen-hidden RMSE는 train-mean 수준(약 4.3 cell)이었다. 짧은 학습에서 예상되는 결과다. 다만 **untrained `init.pkl` feature만으로도** place retrieval recall 0.28(chance 0.055), 전진 가능성 AP 0.94(base rate 0.19)가 나왔다. 따라서 이 두 능력군은 chance가 아니라 untrained 통제와 비교해야 한다.
- **Teacher offline 사전 점검.** 실제 episode 60개를 썼다. 행동 정책은 관측을 무시하는 sticky explore이고, reward는 24회, held-out success step은 384개였다. 이 조건에서 다섯 teacher 변형(hidden 128/16/linear, L2 유무) 중 어느 것도 held-out information gain이 양수가 되지 않았다. 정규화하지 않은 head는 과적합해 shuffled 통제보다 낮아졌다. L2를 준 head는 Δ≈0으로 수렴했다. Gate는 모든 경우 닫힌다. 이 결과는 "과거 rollout을 넣었다는 이유만으로 좋은 target이 생기지 않는다"는 우려를 실제 데이터에서 확인한 것이다. Teacher가 유용해지려면 학습된 정책이 성공을 충분히 자주 만들어야 한다. 그래서 pilot config는 `teacher_l2=0.001`로 둔다. `teacher_only` 단계에서 정보가 생기는 시점을 측정한다.

## 알려진 위험과 한계

- **초기 성공이 드물다.** 학습 초기에는 성공이 드물어 teacher의 BC 신호가 약하다. Gate가 오래 닫혀 있을 수 있다. 이는 결함이 아니라 의도된 동작이다. Gate 증거는 success step 수로 가중하고, 최소 `teacher_min_success_steps`를 요구한다. `teacher_only` 조건으로 언제 정보가 생기는지 따로 측정한다. 위 offline 점검에서는 정보가 생기지 않았다.
- **SDPO형 증류가 이득이 없을 수 있다.** 이론상 SDPO형 증류의 기댓값은 self-imitation과 같다. Teacher가 일반화하지 못하면 이득이 없을 수 있다. 그것도 H2의 유효한 결과다.
- **Matured chunk는 H step 늦다.** 나이는 약 (L+H)/L update다. 엄밀한 on-policy가 아니므로 `feedback_chunk_age`로 기록한다.
- **Teacher가 인과적이지 않을 수 있다.** `actual_success` query는 hindsight라 teacher가 행동할 수 없다. 행동 검증은 counterfactual success query로만 한다.
- **Self-imitation은 목적함수를 바꾼다.** 내부 return은 n-step bootstrapped return이다. Off-policy 보정을 한 policy gradient가 아니다.
- **Simulator와 replay 상태의 정확한 재개는 지원하지 않는다.** 새 output에서 실행한다.
- **로컬 macOS CPU는 약 100 transition/s다.** 연구 규모 학습은 Linux GPU host에서 별도로 확인해야 한다.
