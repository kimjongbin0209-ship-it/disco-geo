# DiscoSpatial

**제한된 reward로 어떤 공간 능력이 형성되는가?** 공개 Disco103 학습 규칙을 recurrent RGB agent에 적용하고, 경험 재사용과 사후 피드백이 능력별로 미치는 효과를 연구합니다.

주 benchmark는 **Memory Maze**입니다. 원래 task의 목표 접촉 +1 보상만 사용하며, 지도·위치·깊이·최단 경로는 actor나 teacher에 제공하지 않습니다. TripoSplat과 기존 물체 인식 실험은 활성 연구에서 분리했습니다. [과거 실험 복구 안내](docs/LEGACY_ARCHIVE.md)

[연구 설계: 공간 능력과 SDPO형 피드백 증류](docs/SDPO_SPATIAL_RESEARCH_PLAN_KO.md) · [연구 프로토콜](docs/RESEARCH_PROTOCOL_KO.md) · [구현·학습 규약](docs/METHODS.md) · [실험 조건표](experiments/conditions.json)

## 현재 구현과 검증 범위

| 항목 | 상태 |
| --- | --- |
| 실제 Memory Maze RGB-only wrapper와 평가 sidecar 분리 | 구현; 로컬 실제 reset/step 확인 |
| 평가용 render telemetry (segmentation visibility, depth 역투영 known cell, coarse depth) | 구현; 재렌더 RGB가 관측과 픽셀 일치, floor 99.8%·wall 99.7% cell 일치 |
| CNN + previous action/reward + GRU + sequential unroll | 구현; online/unroll·reset·history gradient 검사; block 단위 prefix 재구성 |
| 공개 Disco103 / AC / AC+next-policy auxiliary | 구현; 실제 고정 rule을 사용한 learner 검사 |
| Fresh·recent sequence replay·positive-advantage self-imitation | 구현; 실제 benchmark smoke 실행 |
| SDPO형 feedback self-teacher (outcome / outcome+success memory, shuffled 통제, 증거 gate, teacher acting) | **구현, 유용성 미검증**; Bayes posterior 복원 단위 검사, smoke 실행; 고정 explore 정책의 offline 점검에서 held-out 정보 이득 없음(gate 닫힘) |
| 여섯 능력군 suite·navigation(primary 3)·seed 집계 report | 구현; smoke 규모로 실제 데이터에 실행 |
| 새 연구의 성능·창발 주장 | 아직 없음; smoke 실행은 연결 검증이지 연구 결과가 아님 ([요약](reports/development/smoke_summary.json)) |

## 연구 질문과 평가

- **H1:** 동일한 recurrent backbone과 제한된 reward에서 Disco는 AC+aux보다 공간 정보를 잘 보존·활용하는가?
- **H2:** 과거 reward의 단순 재사용을 넘어 feedback-conditioned distillation이 추가 이득을 주는가?
- **H3:** Teacher의 개선 신호가 유용한 사후 feedback 때문인가, 새로운 감독 정보나 추가 연산 때문인가?

능력은 자기 위치/이동 추정, 보이지 않는 물체의 기억, 장소 재인식, 지도·연결 관계, 국소 기하/행동 가능성, 실제 탐색의 여섯 군으로 나눕니다. 주 지표는 **seen-hidden target 위치, causally known wall map, 조건을 맞춘 재방문 효율**입니다. OOD는 별도 능력 개수가 아니라 이 능력들의 일반화 축입니다.

Frozen probe에서 정보가 읽히는 것, 새 controller로 전이하는 것, 동결한 actor가 실제로 활용하는 것을 분리합니다. Layout/episode 단위 split과 seed/layout 단위 통계를 사용하며 frame 수를 독립 표본 수로 세지 않습니다. 계획한 5 seeds는 pilot용이며 confirmatory 표본 수는 power 계획 후 별도로 고정합니다. 전체 visibility telemetry가 없는 generic readout을 주 능력 평가로 부르지 않습니다.

## 모델과 학습

```text
RGB → CNN(SiLU) → signed projection + LayerNorm → image feature
                       + previous action / previous reward
                       → GRU → belief state → 공식 알고리즘별 heads
```

Disco103의 update-rule 가중치는 고정하고 시각 encoder·GRU·policy/prediction/value heads를 학습합니다. Action-model LSTM과 actor GRU는 별개입니다. `is_first`를 저장하고, rollout 시작 hidden state에서 실제 sequential unroll을 수행합니다. LAST는 bootstrap에만 사용하고 원래 continuation discount를 보존합니다.

Recent replay는 raw episode sequence를 저장합니다. 현재 encoder와 recurrent state를 episode 시작부터 재계산하고 원래 behavior logits를 유지합니다. Fresh/replay의 gradient를 한 번에 혼합하고 meta-rule state·EMA는 fresh stream만으로 한 번 갱신합니다. Replay age와 과도한 log-ratio를 제한하며 이 규약 자체를 Disco의 확장으로 기록합니다.

SDPO형 증류는 `L = L_Disco,fresh [+ α·L_replay] + λ_SIL·L_SIL + λ_T·L_teacher + λ_SD·gate·KL(student ‖ stopgrad(teacher_EMA))`입니다. Teacher는 학생 belief(stop-gradient) 위의 residual head `base + Δ(belief, feedback)`이며, 자기 attempt의 reward outcome과 같은 episode·같은 target 색의 이전 성공 frame만 봅니다. 학습된 Δ는 log P(outcome | history, action)을 추정하므로 success query가 soft policy improvement가 되고, 그 기댓값은 self-imitation과 같습니다(H2의 예측). 증류는 shuffled 통제보다 나은 held-out 정보량이 확인될 때만 열립니다. [설계와 근거](docs/SDPO_SPATIAL_RESEARCH_PLAN_KO.md) · [세부 규약](docs/METHODS.md)

## 설치

Python **3.12**를 사용합니다. 다음 환경 stack은 로컬 macOS CPU 연결 검증에 사용했습니다. Linux CUDA와 EGL은 target host에서 별도로 검사합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m disco_spatial --help
python -m disco_spatial protocol --config configs/development.json
```

Memory Maze 1.0.3은 dm_env 경로에서도 import 과정에 Gym을 요구하여 이를 명시적으로 pin했습니다. 실제 실행은 dm_env interface를 사용합니다. 의존성 버전은 `pyproject.toml`에 고정합니다. Memory Maze의 border 구현이 64 미만에서 영상을 덮으므로 실제 환경은 64픽셀 이상을 강제합니다.

macOS 로컬 렌더링 검사:

```bash
MUJOCO_GL=glfw python -m disco_spatial doctor --environment \
  --output runs/environment/doctor.json
```

Linux NVIDIA 서버:

```bash
bash scripts/setup_gpu.sh
source .venv.cuda/bin/activate
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

## 개발용 실행

`integration.json`은 **8 transitions만** 확인하는 연결 검증입니다. Validation이 없어 best checkpoint를 만들지 않습니다.

```bash
python -m disco_spatial train --config configs/integration.json \
  --output runs/integration/disco-fresh
python -m disco_spatial train --config configs/integration.json \
  --recipe recent_replay --output runs/integration/disco-replay
```

더 긴 development run은 config를 명시합니다. 기존 디렉터리는 덮어쓰지 않습니다.

```bash
python -m disco_spatial train --config configs/development.json
python -m disco_spatial train --config configs/replay_development.json
```

Self-imitation과 feedback teacher는 config field 또는 CLI override로 켭니다. 조건표 전체는 `scripts/expand_conditions.py`가 seed별 config로 펼칩니다.

```bash
python -m disco_spatial train --config configs/smoke.json --self-imitation-weight 0.5 \
  --output runs/smoke/disco-sil
python -m disco_spatial train --config configs/smoke.json --teacher outcome_memory \
  --teacher-query counterfactual_success --distillation-weight 0.5 --output runs/smoke/disco-fd-memory
python scripts/expand_conditions.py --stage P1_learning_gate --output runs/pilot/configs
```

실제 전체 에피소드, memory-reset intervention, teacher acting 검증:

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

같은 관측 이력에서 method별 표현을 비교합니다. 수집은 고정된 탐색형 reference policy(`--policy explore`)와 render telemetry를 사용합니다. Validation은 seed 4,000,000대, final test는 5,000,000대에서 별도로 수집합니다.

```bash
python -m disco_spatial collect --partition probe_train --seed 3000000 \
  --episodes 4 --max-steps 1000 --output datasets/probe_train
python -m disco_spatial extract --checkpoint runs/development/disco-seed0/best.pkl \
  --collection datasets/probe_train --output runs/features/probe_train
```

세 partition을 같은 checkpoint로 추출한 뒤 여섯 능력군 suite를 실행합니다. 각 run의 `init.pkl`로 같은 초기화의 untrained 통제도 추출합니다.

```bash
python -m disco_spatial abilities --train runs/features/probe_train \
  --validation runs/features/probe_validation --test runs/features/probe_final_test \
  --output runs/abilities/disco-seed0.json
python -m disco_spatial report --reference acaux \
  --runs acaux:0:runs/abilities/acaux-seed0.json:runs/navigation/acaux-seed0.json \
         disco:0:runs/abilities/disco-seed0.json:runs/navigation/disco-seed0.json \
  --output runs/reports/pilot.json
```

Layout fingerprint는 grid 회전·반사를 동일하게 취급하며 실제 교집합이 있으면 fitting을 거부합니다. 이 검사는 probe partition 간 geometry 분리를 확인합니다. RL train geometry inventory는 아직 연결하지 않아 **RL train과 모든 probe layout의 완전 분리가 입증됐다고 주장하지 않습니다**.

## 재현성과 남은 연구 작업

각 run은 config, protocol hash, source hashes, 환경 버전, learner log, typed raw trajectory, checkpoint를 저장합니다. `latest.pkl`은 정책·optimizer snapshot입니다. **Simulator와 replay의 정확한 재개는 아직 지원하지 않습니다.** 복구 가능한 full-state resume을 구현하기 전에는 새 output에서 실행합니다.

Confirmatory mode는 비활성입니다. 다음 순서는 GPU host의 `P1_learning_gate`(random 이상 학습, memory-reset 하락, teacher 증거) → `P2_pilot_matrix` 5 seeds → power 계획과 design lock입니다. Teacher에 GT map/pose를 제공하는 guided study는 주 실험과 분리합니다.

```bash
python -m pytest -q
ruff check disco_spatial tests
bash -n scripts/setup_gpu.sh
```

## 출처

- [Memory Maze](https://github.com/jurgisp/memory-maze), [논문](https://arxiv.org/abs/2210.13383)
- [DiscoRL](https://github.com/google-deepmind/disco_rl), [고정 source·checkpoint provenance](third_party/PROVENANCE.json)
- [SDPO](https://arxiv.org/abs/2601.20802), [공식 코드](https://github.com/lasgroup/SDPO)

마지막 개발 갱신: 2026-10-28.
