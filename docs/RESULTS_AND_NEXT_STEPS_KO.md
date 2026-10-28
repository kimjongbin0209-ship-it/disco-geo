# DiscoGeo 결과와 상세 실패 원인 분석

분석 갱신: 2026-10-28. [README](../README.md)는 결과 요약, 이 문서는 실패의 구조와 검증 순서, [재현 가이드](REPRODUCING.md)는 실행 방법을 다룹니다.

**범위:** 보존된 본 실험과 소스 코드를 분석했습니다. 문서 작성 중 새 학습이나 3D 생성은 실행하지 않았습니다. 결과 집계 스크립트는 기존 숫자의 일관성을 검증하며, 기존 audit의 기하·adapter 진단을 처음부터 재실행하는 스크립트는 아닙니다. 각 원인을 확정된 구현 사실, 관찰된 현상, 개입이 필요한 원인 가설로 구분합니다.

## 1. 무엇이 성공했고 무엇을 입증하지 못했는가

인식 reward를 이용한 CNN 학습, frozen Disco103 rule 적용, seed별 checkpoint와 feature 보존, 분리된 identity의 기하 probe, frozen TripoSplat conditioning 전이 및 paired control 생성까지 파이프라인은 완료했습니다. 연구 가설인 **“Disco가 더 접근 가능한 기하 표현을 만들고, 그 이점이 생성으로 이어진다”**는 입증되지 않았습니다.

실패는 서로 다른 세 주장으로 나누어야 합니다.

| 검증 대상 | 관찰 | 판단 |
| --- | --- | --- |
| 인식 task를 학습하는가 | Disco test 89.32%, AC+aux 90.23%, 미학습 48.70% | 학습됨. 이 설정의 Disco 고유 우위는 미확립 |
| 기하 정보가 probe에서 더 잘 읽히는가 | Disco depth/pose/relative-pose가 미학습보다 나쁨 | 현재 decoder와 checkpoint에서 개선되지 않음. Depth 목적 불일치로 해석 제한 |
| RL 표현이 생성에 추가 이득을 주는가 | Adapter가 teacher 평균보다 나쁘고, Disco 생성 우위 없음 | Downstream fitting 병목과 미정렬 평가로 encoder 원인 분리 불가 |

Disco가 모든 환경에서 기하를 학습하지 못한다는 결론, 기하 정보가 최종 feature에 전혀 없다는 결론, zero conditioning이 가장 좋은 복원을 만든다는 결론은 현재 결과로 뒷받침되지 않습니다.

## 2. 결과와 데이터 범위

512 identities × 12 views × 64×64 RGB이며 split은 RL train 256 / RL validation 51 / probe train 103 / probe validation 51 / test 51입니다. 미학습 포함 6조건을 RL seed 0·1·2에서 비교했습니다. 학습 조건은 각각 약 500k 유효 transitions입니다. Depth/absolute-pose probe의 요청 label budget은 5,000이지만 실제 사용량은 103×12=1,236 images입니다. Relative-pose의 5,000은 view pair sampling 수이므로 같은 단위가 아닙니다.

Conditioning cache는 128장(train 89/validation 19/test 20)이고, 최종 생성은 test 20 중 선택한 4개 물체에서 이루어졌습니다. 18개 encoder × 4 inputs × learned/zero/shuffled와 공통 native 4를 합해 220개입니다. Noise seed는 42 하나, adapter는 30 epochs입니다. [Pipeline](../runs/full-experiment/pipeline.json), [cache manifest](../datasets/tripo_cache/manifest.json), [transfer plan](../runs/full-experiment/transfer/plan.json)

### 인식과 기하 결과

±는 3개 RL seed 간 표본 표준편차이며 신뢰구간이 아닙니다. 정책 test는 선택된 best checkpoint의 greedy 256 episodes입니다.

| 방법 | 인식 성공률 ↑ | 평균 이동 | Depth RMSE ↓ | Absolute yaw ↓ | Relative yaw ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 미학습 CNN | 48.70 ± 3.52% | 0.833 | 0.4848 ± 0.0119 | 47.38 ± 0.81° | 48.86 ± 1.44° |
| AC | 81.64 ± 9.60% | 0.167 | 0.5374 ± 0.0568 | 66.91 ± 6.35° | 60.02 ± 4.50° |
| AC+aux | 90.23 ± 0.78% | 0.176 | 0.5332 ± 0.0185 | 66.37 ± 2.11° | 55.41 ± 1.46° |
| Disco103 | 89.32 ± 1.58% | 0.000 | 0.5631 ± 0.0205 | 65.64 ± 5.58° | 67.25 ± 9.43° |
| Disco−aux | 88.15 ± 0.60% | 0.000 | 0.5507 ± 0.0390 | 57.19 ± 5.80° | 68.85 ± 7.88° |
| Disco−yz | 64.32 ± 21.76% | 0.000 | 0.5147 ± 0.0238 | 51.23 ± 9.87° | 50.16 ± 5.01° |

[원본 summary](../runs/full-experiment/study/report/summary.json), [seed별 CSV](../results/full-experiment/rl_metrics.csv)

### 최종 생성 결과

아래는 **UNREGISTERED orbit diagnostics**입니다. Input view를 제외한 11개 view를 image별로 평균하고, images를 RL seed별로 평균한 뒤 seed 통계를 계산합니다. 실제 기하 복원 순위로 해석하지 않습니다.

| 조건 | PSNR ↑ | Foreground PSNR ↑ | Silhouette IoU ↑ |
| --- | ---: | ---: | ---: |
| Native RGB | 10.132 | 9.191 | 0.3971 |
| Zero conditioning | 11.324 | 9.249 | 0.4928 |
| 미학습 CNN → adapter | 11.243 ± 0.187 | 8.852 ± 0.094 | 0.4833 ± 0.0050 |
| AC → adapter | 10.922 ± 0.114 | 8.674 ± 0.143 | 0.4896 ± 0.0026 |
| AC+aux → adapter | 10.820 ± 0.143 | 8.454 ± 0.263 | 0.4776 ± 0.0065 |
| Disco → adapter | 11.206 ± 0.144 | 8.543 ± 0.322 | 0.4739 ± 0.0262 |
| Disco shuffled | 10.782 ± 0.233 | 8.149 ± 0.200 | 0.4291 ± 0.0213 |

[전체 19개 조건](../results/full-experiment/transfer_metrics.csv), [물체별 결과](../runs/full-experiment/transfer/report/per_image.csv). Native는 shared baseline입니다. 동일 noise와 조건의 zero 반복은 여러 독립 seed의 증거가 아닙니다.

## 3. Task가 다중 시점 가설을 충분히 요구하지 않았다

**확인된 구현:** 첫 step부터 SAME/DIFFERENT를 선택할 수 있습니다. DIFFERENT는 임의의 다른 identity이고 current/reference view는 독립적인 uniform sampling입니다. 다른 방향에서만 드러나는 차이나 초기 silhouette가 유사한 hard negative는 구성하지 않습니다. SAME pair에서 두 view가 같은 확률은 1/12이고, 전체 episode의 기대 약 4.17%는 같은 object/view 쌍입니다. 이 subset만으로 89% 성공을 설명할 수는 없지만 쉬운 경우가 포함된다는 사실은 확인됩니다. [ring.py](../disco_geo/envs/ring.py)

Reward는 정답+1, 오답/timeout0, 이동당−0.02입니다. 추가 관측이 개선할 정답 확률보다 비용이 크면 즉시 답하는 것이 합리적일 수 있습니다. 예를 들어 한 번 움직여 얻는 정확도 증가가 2%p보다 작다면, 할인 등을 제외한 단순 기대 return에서는 비용을 상쇄하지 못합니다. 이것은 보상 구조의 설명이며 현재 policy의 실제 정보 이득을 측정한 값은 아닙니다.

**관찰:** Disco 세 seed의 best checkpoint가 greedy test256 episodes에서 평균 이동 0으로 성공률 87.89/89.06/91.02%를 얻었습니다. 이 정책은 초기 current/reference RGB만으로 높은 정확도를 얻었습니다. 학습 actor는 softmax로 action을 샘플링하고 평가 actor는 argmax를 사용합니다. 저장된 학습 로그에는 action histogram과 entropy가 없어 학습 내내 이동하지 않았는지 확인할 수 없습니다. [runner.py](../disco_geo/training/runner.py), [upstream actor](../third_party/disco_rl/disco_rl/agent.py)

**가능성이 높은 설명:** 현재 pair의 외관·silhouette 통계만으로 reward를 충분히 얻어 정보 수집을 학습할 압력이 약했을 수 있습니다. Audit에서 Disco PC1과 foreground area의 절대 상관이 0.650–0.812인 것은 이 가설과 부합하지만 정책이 area를 실제 결정 원인으로 사용한다는 증거는 아닙니다.

**검증:** 고정된 episode bank에서 no-move, 강제 1회전, random 탐색, oracle-view 선택, stochastic/greedy policy를 비교합니다. SAME/DIFFERENT별 confusion matrix, 첫 step 답변 비율, action 확률, 이동별 정확도를 저장합니다. Move cost sweep은 validation에서 수행합니다. 초기 view가 유사하고 회전 후에 구별되는 hard pair로 추가 관측의 이득을 확인해야 합니다. 이동 횟수를 강제로 늘리는 것만으로 능동 인식 성공을 판정하지 않습니다.

## 4. 관측 이력을 결합하는 memory가 없다

**확인된 구현:** CNN은 current/reference RGB를 처리하고 두 feature, 차이, 곱을 이어 붙여 policy torso에 넣습니다. `should_reset`은 버립니다. Action-model LSTM은 현재 torso에서 root embedding을 생성하므로 환경 시간축의 관측 이력을 운반하는 recurrent actor가 아닙니다. [encoder.py](../disco_geo/models/encoder.py), [action_models.py](../third_party/disco_rl/disco_rl/networks/action_models.py)

이동으로 더 유용한 현재 view를 선택하는 행동은 가능하지만, 서로 다른 시점의 누적 증거를 기억하여 결합하는 구조는 없습니다. 따라서 task를 더 모호하게 만드는 것만으로는 해결되지 않을 수 있습니다.

**검증:** `쉬운 pair / hard pair × feedforward / recurrent memory`의 2×2 실험을 먼저 하고, 같은 memory 구조에서 AC+aux와 Disco를 비교합니다. Hard-pair만, memory만, 둘 다의 효과를 분리해야 task 난이도와 architecture 변경을 혼동하지 않습니다.

## 5. 최종 feature가 인식에 맞춰 압축되었다

**관찰:** Probe-train 103 identities × 12 views에서 미학습 CNN의 raw effective rank는 28.51–36.42, Disco는 2.17–4.22입니다. Disco projection 256차원 중 175–198차원은 이 1,236장 전체에서 0입니다. AC도 effective rank 2.14–3.12로 낮아져 Disco만의 현상은 아닙니다. [Audit](../runs/full-experiment/audit/diagnostics.json)

Effective rank는 covariance spectrum의 엔트로피를 지수화한 값입니다: `exp(−Σ pᵢ log pᵢ)`, `pᵢ = σᵢ² / Σ σⱼ²`. 정확한 선형 rank나 비선형 정보량이 아닙니다. 표본에서 항상 0인 차원을 모든 가능한 입력에서 영구히 죽은 neuron이라고 단정할 수도 없습니다.

Encoder는 stride 2 convolution → Flatten → Linear 256 → ReLU입니다. Global average pooling을 사용하지 않습니다. Probe와 adapter는 이 최종 projection을 사용하며 convolution map과 pre-ReLU feature를 비교하지 않았습니다. 따라서 현재 결과는 선택된 최종 feature와 작은 decoder에서의 접근 가능성에 대한 결과입니다. [encoder.py](../disco_geo/models/encoder.py), [features.py](../disco_geo/evaluation/features.py)

**원인 가설:** 인식에 충분한 저차원 단서를 강화하면서 기하를 읽는 데 필요한 variation이 약해졌을 수 있습니다. ReLU 활성 소실, feature scale 증가, task에 무관한 정보 제거가 후보지만 어느 하나의 인과적 기여도는 확인하지 못했습니다.

**검증:** 같은 고정 이미지에서 conv/pre-ReLU/post-ReLU layer별 probe, effective rank, feature norm을 checkpoint의 transition 수와 함께 추적합니다. 그 뒤 activation/normalization 변경을 개입 실험으로 평가합니다. 단순히 feature dimension을 늘리는 것을 해결책으로 확정하지 않습니다.

## 6. Depth probe의 목적 불일치와 쉬운 깊이 분포

**확인된 구현:** Depth는 4픽셀 간격으로 downsample한 16×16 grid입니다. Ridge fitting target에는 background=0이 포함됩니다. Alpha 선택과 test의 RMSE/AbsRel/δ는 `target>0` foreground만 사용합니다. [probes.py](../disco_geo/evaluation/probes.py)

배경을 0으로 둔 무제약 dense MSE의 pixel별 최적해는 이론적으로 `P(foreground | feature) × E(depth | feature, foreground)`입니다. 현재의 유한한 linear ridge가 이 해를 그대로 달성한다는 뜻은 아닙니다. Foreground 위치를 불확실하게 예측하면 출력 깊이가 낮아지므로, 실제 foreground에서만 점수를 계산할 때 불리할 수 있습니다. 현재 오차 중 이 메커니즘의 기여량은 측정하지 않았습니다. Validation에서 alpha를 고르는 것만으로 학습 목적의 차이가 없어지지는 않습니다.

**관찰:** Probe-train foreground 비중은 35.03%, 평균 깊이는 1.83968입니다. 이 train 평균을 모든 pixel에 출력하는 상수의 test 결과는 RMSE 0.20020, AbsRel 0.08830, δ1.25=0.96793입니다. 미학습 RMSE 0.48478, Disco 0.56307보다 좋습니다. 이 baseline은 평가용 GT mask로 scoring할 뿐 입력 이미지에서 mask를 받아 예측하는 모델이 아닙니다.

**분포의 제약:** 생성기는 object 최대 span을 1로 정규화하고 camera 반경·고도·FOV·조명을 고정합니다. 깊이 범위가 제한되어 평균 예측이 강할 수 있습니다. Absolute yaw도 world lighting의 shading으로 유추할 수 있으므로 pose 점수 자체가 3D 구조의 증거는 아닙니다. [procedural.py](../disco_geo/data/procedural.py)

**검증:** Foreground-aware fitting과 silhouette/depth 분리 probe를 추가하고, 상수·pixelwise mean·downsampled RGB·silhouette-only baseline을 같은 label budget으로 비교합니다. 알려진 기하 feature를 넣는 positive control로 decoder가 정상인지 검증합니다. 이후 camera distance/elevation·lighting/background를 변화시킨 held-out domain을 평가합니다. Geometry label은 계속 사후 probe에서만 사용하고 RL에는 넣지 않습니다.

## 7. Adapter가 표현 가능한 평균 baseline도 회수하지 못했다

이 부분이 생성 실패에서 가장 직접적으로 확인된 downstream 병목입니다. 아래의 상세 audit는 seed 0의 선택된 세 방법과 validation 19장 범위이며 모든 seed로 일반화하지 않습니다. Teacher 평균 baseline은 scalar 평균이 아니라 train 89장의 teacher target을 이미지 축으로 평균한 `[4101,1408]` token/channel별 상수 tensor이며 입력 이미지와 무관합니다.

| Seed 0 조건 | Validation combined MSE ↓ |
| --- | ---: |
| Train teacher 평균 상수 | **0.421100** |
| 미학습 CNN adapter | 0.538923 |
| Disco adapter | 0.538473 |
| AC+aux adapter | 0.543462 |

Disco train full-token MSE도 0.527345로 teacher 평균 0.409337보다 나쁩니다. **현재 모델은 `predict=0`, `bias=train_teacher_mean`으로 이 상수 해를 표현할 수 있습니다.** 따라서 상수보다 나쁜 fitting을 encoder의 정보 부족이나 rank 8 제약만으로 설명할 수 없습니다. 적어도 현재 최적화·초기화·budget이 더 좋은 feasible solution을 회수하지 못한 것이 확인됩니다. [Audit](../runs/full-experiment/audit/diagnostics.json), [adapter.py](../disco_geo/generative/adapter.py)

### 구조와 손실

출력은 `token_basis @ image_code + token_bias`입니다. Target은 feature1 `[1,4101,1280]`, feature2 `[1,4101,128]`입니다. Image-dependent residual의 token×channel matrix rank는 최대 8이며, shared bias는 별도로 존재합니다.

| 구성 | Parameter 수 |
| --- | ---: |
| Per-token bias | 5,774,208 |
| Token basis | 32,808 |
| 첫 Linear | 65,792 |
| 마지막 Linear | 2,894,848 |
| 합계 | 8,767,656 |

Bias가 전체의 65.86%이고 0으로 초기화됩니다. Train 89장에 비해 모델이 크지만, train도 평균보다 나쁘므로 현재 관찰을 단순한 과적합으로 설명하기는 어렵습니다. 반대로 더 오래 학습하면 encoder 간 우위가 생긴다는 보장도 없습니다.

Loss는 두 head를 concatenate한 단일 MSE입니다. 각 element의 가중치는 같고, head별 총 element 수에 따른 loss 기여 가중치는 feature1이 1280/1408=90.91%, feature2가 9.09%입니다. Validation head1 MSE는 Disco 0.55004 대 teacher 평균 0.42143이고, head2는 0.42277 대 0.41781입니다. 큰 head1의 fitting 격차가 combined 오차를 지배합니다. 입력 RL feature는 train-only 표준화 없이 전달됩니다. [train_adapter.py](../disco_geo/generative/train_adapter.py)

### Budget과 token sampling

89 images × 30 epochs = 2,670 item optimizer steps입니다. 각 step은 4,101 token 중 128개, 약 3.12%를 샘플링합니다. 한 image-token pair의 기대 직접 관측 횟수는 `30×128/4101 = 0.9364`회이며, 한 번도 샘플되지 않을 확률은 `(1−128/4101)^30 ≈ 38.62%`입니다. 이 값은 sampling 방식에서 계산한 기대치입니다.

여기서 38.62%는 **image-token 쌍**에 대한 값입니다. 전체 train set을 합친 token당 기대 직접 관측은 `89×30×128/4101 = 83.34`개입니다. “전체 token의 38.6%가 전혀 학습되지 않았다”는 뜻이 아닙니다. AdamW의 momentum과 decay도 있으므로 이를 parameter update 횟수로 부르지 않습니다.

모든 18개 adapter의 best epoch는 마지막 30입니다. 마지막까지 validation이 개선되는 로그와 train/validation 모두 평균보다 높은 오차는 현재 fitting이 미완성이라는 해석을 강하게 지지합니다. Sparse token supervision, 큰 zero-initialized bias, feature scale 차이와 rank는 후보 원인이지만 기여도는 분리하지 못했습니다.

### 입력을 완전히 무시한 것도 아니다

Validation 19장의 각 token/channel 좌표에서 이미지 축 population variance(ddof=0)를 계산해 token/channel 전체에 평균한 Disco prediction variance는 teacher 대비 feature1의 4.92%, feature2의 1.06%입니다. 변화량이 작지만 정상 입력의 MSE가 shuffled/mean 입력보다 낮으므로 입력을 완전히 무시한다고 말할 수는 없습니다. 여기서 mean-input control은 평균 RL feature를 기존 adapter에 넣는 것이며, teacher 평균 tensor를 직접 출력하는 상수 baseline과 다릅니다. 미학습 CNN에서도 같은 방향이 관찰되어 그 차이를 RL 특유의 기하 정보로 연결할 수 없습니다.

**검증:** Teacher 평균 bias 초기화+residual fitting, train-only feature 표준화, 작은 linear/PCA-ridge baseline, 소수 이미지 overfit 검사를 순서대로 합니다. 각 head의 validation MSE·평균 대비 explained error reduction·입력별 variance를 보고 충분한 수렴을 확인합니다. 통과 기준은 held-out에서 teacher 평균을 이기고 입력별 변화가 의미 있게 유지되는 것입니다. Rank와 token budget은 이 기본 해를 확보한 뒤 비교합니다. Encoder와 generator는 계속 frozen으로 유지합니다.

## 8. 생성 평가의 좌표·camera 검증이 끝나지 않았다

**확인된 구현:** Evaluator는 생성 NPZ의 내부 좌표를 읽고 benchmark absolute yaw로 렌더합니다. Source view는 평가에서 제외하지만 source-yaw offset이나 predicted camera 보정에는 사용하지 않습니다. Generator가 저장한 `sampled_camera.npy`도 evaluator가 읽지 않습니다. 정렬 transform을 제공하지 않은 결과는 명시적으로 unregistered입니다. [evaluate.py](../disco_geo/generative/evaluate.py), [generate.py](../disco_geo/generative/generate.py)

PLY/SPLAT export에는 upstream viewer축 변환이 있으나 그것이 analytic benchmark frame과 일치한다는 검증은 없습니다. CPU renderer는 합성 benchmark와 같은 camera basis·elevation 0.25·FOV 35·배경 235를 사용합니다. 따라서 현재 근거로 합성 camera 구현 자체에 명백한 버그가 있다고 말할 수는 없습니다. 확인되지 않은 부분은 생성 frame→benchmark frame, 입력의 crop/camera convention, 공식 rasterizer와의 수치 parity입니다. [render.py](../disco_geo/generative/render.py)

Native teacher 입력은 background removal과 crop/resize를 거치고, RL feature는 원래 64×64 RGB에서 나옵니다. 이 preprocessing 차이와 camera/crop 관계도 점검 대상입니다. Teacher의 stochastic encoding이 만드는 noise floor는 가능한 가설이지만 크기를 측정하지 않았습니다.

**관찰:** Scene 49 예시에서 native는 L형을 어느 정도 보존하고 learned 조건은 비슷한 막대처럼 보입니다. Zero는 다른 유색 형상을 만들지만 전체 diagnostic PSNR/IoU가 높습니다. 한 예시로 전체 현상을 단정할 수는 없지만, 현재 점수의 순위를 실제 geometry 우위로 해석하면 안 된다는 근거입니다.

![Scene 49 seed0의 미정렬 정성 비교](../runs/full-experiment/transfer/report/qualitative_scene49_seed0.png)

**검증:** Validation 물체에서 source/novel-view camera convention과 좌표 변환을 확인하고 analytic primitive 및 공식 renderer와의 sanity check를 합니다. Test view를 보며 transform을 최적화하지 않고 고정된 절차로 정렬합니다. Native·mean conditioning·zero의 정성/정량 관계가 해석 가능한지 먼저 확인한 뒤 learned 비교를 확대합니다. Native가 반드시 모든 물체에서 zero를 이겨야 한다는 이론적 보장을 가정하지는 않습니다.

## 9. 비교 설계와 표본 수의 한계

**Checkpoint 선택:** Best는 RL-validation return만으로 선택합니다. Geometry label로 선택하는 leakage는 피하지만 geometry 보존을 최적화하는 기준은 아닙니다. `no_yz_loss` seed 0의 best는 update 1, 거의 초기 모델입니다. 해당 조건이 probe에서 미학습에 가까운 것을 yz 제거의 기하 개선으로 해석하면 selection 차이가 섞입니다. [runner.py](../disco_geo/training/runner.py), [audit](../runs/full-experiment/audit/diagnostics.json)

**Loss perturbation:** `no_yz_loss`는 y/z fitting loss의 weight를 제거하지만 그 prediction을 입력으로 사용하는 frozen meta-rule은 남습니다. `no_aux_policy`도 공개 rule의 학습 분포를 교란하는 perturbation입니다. 이를 새 rule을 meta-learning한 결과나 특정 loss의 순수한 geometry 기여로 해석할 수 없습니다. [upstream Disco loss](../third_party/disco_rl/disco_rl/update_rules/disco.py)

**Trajectory와 budget:** 초기 CNN과 nominal transition 수는 맞지만 stochastic policy가 달라 방문 이미지·episode 길이·learner update 수가 다릅니다. 매 1,000 updates snapshot은 같은 transition 경험을 뜻하지 않습니다. 현재 비교는 각 알고리즘을 이 설정에서 실행한 결과입니다. 공통 behavior-policy trajectory 실험은 별도 off-policy 조건으로 명시하고 알고리즘의 importance correction과 호환성을 확인해야 합니다. 대칭적인 작은 hyperparameter sweep도 이후 필요합니다.

**Shuffled control:** Wrong-scene pool은 test identity 1개, RL-train identity 2개, adapter-train identity 1개로 섞여 있습니다. 정상 input 4개는 모두 held-out test identities이므로, held-out 물체들 안에서 identity만 순열한 엄밀한 control은 아닙니다. 단순히 cache 밖이라는 사실만으로 차이를 설명할 수는 없지만, 학습 노출과 split 분포가 다르다는 confound는 남습니다. 같은 split·시점 분포 안의 다른 identity로 derangement를 만들고 mean-condition/bias-only/train-feature-permutation control을 추가해야 합니다.

**반복 수:** RL seed 3개는 기술통계를 제공하며 작은 차이의 통계적 신뢰성을 확립하지 못합니다. Random-policy control은 같은 평가 RNG의 결과를 seed별로 반복하므로 SD 0을 안정성의 증거로 삼지 않습니다. 생성은 test 4개·noise seed 1개로 domain과 sampling variance를 충분히 평가하지 못합니다.

## 10. 다음 실행 순서와 통과 기준

| 순서 | 작업 | 통과 기준 |
| --- | --- | --- |
| 1 | 기존 feature로 foreground-aware probe, 상수/RGB/silhouette baseline, positive control | Decoder가 알려진 geometry를 읽고 baseline 대비 점수를 해석할 수 있음 |
| 2 | Teacher-mean residual adapter, feature 표준화, 작은 baseline, 충분한 수렴 | Held-out MSE가 teacher 평균보다 낮고 입력별 variation을 회수함 |
| 3 | Hard pair와 recurrent memory의 2×2 비교, 고정 episode bank | 같은 observation budget에서 추가 view가 정확도·return을 개선함 |
| 4 | 같은 구조의 AC+aux/Disco, layer별 probe·rank·best/latest·transition-matched 비교 | 인식과 기하 접근성의 차이가 선택 기준·노출 데이터에 얼마나 의존하는지 분리 |
| 5 | Validation camera/frame/render calibration 후 물체·noise seed 확대 | Held-out paired 생성 차이를 정성·정량에서 일관되게 해석 가능 |

1과 2는 저장된 데이터·feature를 재사용하고, 3의 task 설계와 5의 renderer calibration은 병행할 수 있습니다. 모델 선택은 train/validation에서 수행합니다. 기존 test를 이미 분석에 사용했으므로 재설계한 최종 주장은 새로운 생성 seed의 미관찰 test identities에서 확인합니다.

후속 연구는 **“다중 시점의 필요성이 인식 reward로 학습한 표현의 기하 접근성을 어떻게 바꾸며, 그 효과가 update rule에 얼마나 의존하는가?”**로 구성하는 것이 현재 관찰과 잘 맞습니다. 인식 향상·표현 압축·기하 접근성을 함께 추적하고, 기하 표현 비교가 성립한 뒤 generative transfer를 연결합니다.

## 11. 결과 보존과 이번 검증의 범위

[집계 스크립트](../scripts/summarize_results.py)는 Python 표준 라이브러리만 사용합니다. [metrics.json](../results/full-experiment/metrics.json)에는 원본 RL/transfer summary와 audit를 담고, [provenance.json](../results/full-experiment/provenance.json)은 입력 SHA-256과 숫자 교차검증을 기록합니다. RL 51행, transfer 57행(19개 조건)의 CSV를 생성하며 원본 orbit 2,420행→image 220행→seed 집계를 확인합니다.

기존 [audit diagnostics](../runs/full-experiment/audit/diagnostics.json)와 [진단 그림](../runs/full-experiment/audit/audit_summary.png)은 당시의 사후 분석 자료입니다. 포함된 CPU test 17개 통과 기록은 역사적 기록이며 이번 문서 작성에서 17개를 다시 실행한 것은 아닙니다. 수치 재집계 통과는 raw checkpoint가 모든 과학적 가정을 만족한다는 보증이 아닙니다.

```bash
python3 scripts/summarize_results.py \
  --run runs/full-experiment --output results/full-experiment
```

공개 구현과 적용 범위: [DiscoRL](https://github.com/google-deepmind/disco_rl), [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat), [고정한 upstream provenance](../third_party/PROVENANCE.json).
