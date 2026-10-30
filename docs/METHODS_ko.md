# 구현 규약과 검증 기준

[English](METHODS.md) | **한국어**

## 수집과 recurrent state

환경은 원래 Memory Maze의 RGB와 0/+1 reward만 제공한다. 목표 색은 원래 이미지 테두리에 포함된다. previous action/reward는 이미 발생한 causal 정보다. FIRST에서 hidden state와 이전 action/reward를 초기화한다. Actor GRU, action-conditioned prediction LSTM, frozen update-rule LSTM은 서로 다르다.

Collector는 실제 환경 action만 transition으로 센다. Batch 중 하나라도 LAST이면 chunk를 끝내고 해당 관측은 bootstrap으로만 처리한다. 다음 chunk 전에 종료한 env만 reset한다. Bootstrap에서 계산한 hidden state는 collector carry에 반영하지 않아 다음 chunk의 관측을 두 번 소비하지 않는다. Time-limit LAST의 continuation=1과 true termination=0을 구분한다.

Online carry는 실제 collector의 truncated-BPTT state다. 새로운 policy parameter로 episode prefix 전체를 매번 재실행하는 정책과 완전히 같은 조건이라고 주장하지 않는다. Replay는 current parameter로 full raw prefix를 재실행해 loss 구간 시작 state를 얻는다.

## Recent replay

Replay sampler는 완전한 episode 또는 FIRST부터 연장되는 불변 raw prefix만 받는다. T actions에 T+1 RGB가 대응한다. Full behavior logits, 원래 reward, policy version, goal counter, termination/truncation을 보존한다. 평가 labels는 별도 sidecar다.

같은 pre-update parameter와 복사된 meta-state에서 fresh/replay loss를 각각 정규화하고 `(1−alpha)grad_fresh + alpha grad_replay`로 한 번 업데이트한다. Meta-RNN, advantage/TD EMA, target network EMA는 fresh branch의 변경만 commit한다. alpha=0은 baseline update와 동일하다. 이 fresh-only lifetime-state 규약은 추가한 학습 전략이다.

개발 기본 replay age는20 updates, 선택 action의 절대 log-ratio 한도는5다. 이 값은 최적값이 아니라 검증할 development setting이다. 과도한 drift/비유한 진단에서는 fresh update로 fallback한다. 보고용 rho는 수치 안정성을 위해 log-ratio±20으로 제한하지만 실제 learner에는 원래 behavior logits를 전달한다. 현재 explicit KL/ESS threshold gate는 없고 해당 값은 로그로 남긴다.

AC는 V-trace, Disco Q는 Retrace형 correction을 사용한다. 이 사실만으로 frozen rule의 policy/y/z targets 전체가 오래된 replay에서 unbiased하거나 안정적이라고 보장할 수 없다. Raw 경험을 재사용하는 것과 당시 feature/meta-target을 재사용하는 것을 구분한다.

`env_processes=true`이면 각 환경이 별도 worker 프로세스(spawn)에서 실행된다. Batch는 모든 명령을 먼저 보낸 뒤 결과를 받는다. Worker의 timestep과 bookkeeping 속성은 in-process 실행과 같으며, 학습 log가 일치하는지 test로 확인한다. Worker 오류는 traceback과 함께 학습 프로세스로 전달된다. 같은 episode 위치의 prefix 재구성(Disco EMA target state, feedback chunk)은 batch 축으로 한 번에 계산한다.

Prefix 재구성은 256/64/16/4/1 크기의 block으로 나눠 unroll한다. Compile 횟수는 제한되고 recurrence는 step 단위 처리와 같다(300 step에서 atol 2e-5로 검사). Replay 적격성 검사는 vectorized되어 있다. Policy version이 episode 안에서 단조이므로, window의 가장 오래된 version은 시작점에, 가장 새 version은 끝점에 있다.

## Self-imitation

`self_imitation_weight > 0`이면 매 update마다 replay에서 reward를 포함한 chunk를 하나 뽑는다(성공 선택, `self_imitation_max_age` 이내). 그 chunk에 positive-advantage 모방 손실을 더한다. Return은 chunk 안의 실제 reward와, chunk 끝에서 현재 value로 bootstrap한 n-step return이다. Baseline V는 rule 자신의 categorical value다. Disco는 Σπq, AC는 v를 쓰고 stop-gradient한다. 목적함수를 바꾸는 방법이며 off-policy 보정을 한 policy gradient가 아니다.

## SDPO형 feedback self-teacher

설계 근거는 [연구 설계](SDPO_SPATIAL_RESEARCH_PLAN_ko.md)에 정리했다. 구현 규약은 다음과 같다.

- **Teacher.** Teacher는 `teacher_*` parameter를 가진 residual head다. Logits는 `base + Δ(stopgrad(belief), feedback)`다. 출력층이 0으로 초기화되어 처음 teacher는 base와 같다. Backbone parameter, optimizer moment, meta-state는 teacher가 없는 agent와 같은 RNG split으로 초기화된다(test로 확인).
- **Feedback.** `outcome`은 H step 안에 +1 reward가 오는지를 나타낸다. 관측되지 않은 window는 censored로 mask한다. 끝난 episode의 미달은 실제 실패로 센다. `success_memory`는 RGB 테두리 색으로 읽은 현재 target과 같은 색의 직전 성공 frame(접근 장면)과 경과 step이다. 성공 다음 관측부터 쓸 수 있고 인과적이다.
- **데이터.** `MaturedChunks`는 각 env의 최신 chunk를 outcome window 전체가 관측된 뒤 한 번만 내보낸다. 첫 reward가 아니라 window 전체를 기다리므로 성공이 실패보다 일찍 풀리는 편향이 없다. Prefix state는 현재 parameter로 재구성한다. Behavior logits는 원래 값을 그대로 쓴다.
- **손실.** Teacher BC는 known step에서 behavior logits를 base로 계산한다(λ_T). 증류는 `KL(student ‖ stopgrad(stopgrad(student) + Δ_EMA(query)))`이다(λ_SD·gate). `actual_success`는 실제 성공 step에만 증류하고 hindsight다. `counterfactual_success`는 query outcome을 1로 고정해 모든 step에 증류하며 인과적이다. Shuffle 통제는 teacher 입력만 (time, batch) 사이에 섞는다. Mask와 return은 각 step에 그대로 둔다.
- **한 번의 update.** Rule gradient(fresh, 또는 replay와 혼합)와 auxiliary gradient를 더해 optimizer를 한 번 적용한다. Rule meta-state는 fresh로만 갱신한다. Auxiliary batch가 없으면 기존 `learner_step_mixed`와 비트 단위로 같다.
- **Gate와 검증.** `TeacherGate`는 held-out 진단을 그 진단이 설명하는 step 수로 가중해 누적한다. Information gain과 shuffled gain은 success step 수로, local improvement와 copy rate는 known step 수로 가중하고, update마다 0.98로 감쇠한다. 효과적인 success step이 `teacher_min_success_steps` 이상이고 네 기준을 모두 넘으면 증류를 연다. 성공이 없는 chunk가 성공 증거를 희석하지 않는다. Evaluation 때마다 인과적 teacher가 같은 validation seed에서 직접 행동한 결과(`teacher_validation`)도 기록한다.
- **용량과 정규화.** `teacher_hidden=0`이면 Δ가 [belief, feedback]과 outcome 상호작용의 선형 함수가 된다. `teacher_l2`는 head parameter에 L2 penalty를 준다. 실제 explore episode의 offline 점검(`scripts/validate_teacher_offline.py`)에서 정규화하지 않은 head는 held-out 성능이 shuffled 통제보다 낮아졌다. L2를 준 head는 Δ≈0으로 수렴했다. 그래서 pilot 기본값은 `teacher_l2=0.001`이다.

Teacher 입력에 simulator state, map, pose, target 위치는 없다. Protocol validator는 teacher 입력 whitelist와 `validated=false`를 강제한다. `FeedbackMemory`(guarded storage)는 legacy API로 남아 있다. 학습 경로는 `disco_spatial.feedback`을 쓴다.

## 평가와 연구 gate

Primary 지표는 세 가지다. Seen-hidden target 위치, causally known wall map, 매칭된 재방문 효율이다. Evaluation 환경은 `telemetry=True`일 때 actor 카메라를 다시 렌더링한다. Segmentation으로 target 픽셀 수를 세고, depth 역투영으로 관측 cell을 구하고, 8×8 depth를 기록한다. 같은 렌더러의 RGB는 관측과 픽셀 단위로 일치한다. Doctor와 collection manifest는 floor 픽셀이 wall cell에 떨어진 비율을 oracle 자기 점검으로 기록한다. Ability suite(`abilities`)와 navigation(`navigate`), seed 집계(`report`)는 구현되어 있다. 아직 연구 규모로 실행한 적은 없다.

Generic feature extraction은 label을 모델에 전달하지 않는다. Causal raw history에서 image feature와 GRU belief를 추출한 다음 readout 전용 artifact에 평가 label을 결합한다. 원본 learner trajectory schema는 이런 artifact를 입력으로 받지 않는다. 위치 label은 episode 시작 위치/방향 기준으로 고정하고 test label로 정렬하지 않는다.

Checkpoint는 원 task validation return으로 선택한다. 여러 ability 중 가장 좋은 test checkpoint를 고르지 않는다. 정책 성능과 frozen readout 성능, 공통-history와 policy가 직접 만든 trajectory의 분포 차이를 구분한다. 전체 통계 규약은 [연구 프로토콜](RESEARCH_PROTOCOL_ko.md)을 따른다.

## 현재 확인하지 않은 것

- 새 benchmark에서 충분한 학습과 memory의 인과적 성능 이득
- 연구 규모의 여섯 능력군 측정과 RL-train geometry inventory
- SDPO teacher의 실제 유용성. Gate가 열리는지, teacher가 직접 행동했을 때 student보다 나은지
- Linux GPU 서버에서 장기 학습의 안정성, VRAM과 예산(처리 속도는 측정함, README 참고)
- Simulator·replay까지 포함한 exact resume

작은 integration 결과는 코드 연결에 대한 증거이며 새로운 공간 능력의 창발 결과가 아니다.
