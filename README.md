# DiscoGeo

DiscoGeo는 **RGB 인식 보상으로 학습한 표현에서 기하 정보를 더 쉽게 읽을 수 있는지**, 그리고 그 표현을 frozen 3D 생성 모델의 conditioning으로 전이할 수 있는지 검증하는 연구 코드입니다. 공개된 Disco103 update rule을 실제로 적용하며, AC·AC+aux·미학습 CNN과 비교합니다.

**현재 결과:** 인식 학습과 전체 GPU 파이프라인은 완료했습니다. 이 설정에서 Disco 고유의 기하 표현 또는 3D 생성 우위는 확인하지 못했습니다. Task 설계가 다중 시점 정보 수집을 강제하지 않았고 평가에서는 초기 RGB 쌍만으로 높은 정확도를 얻었습니다. Depth probe·conditioning adapter·3D 좌표 정렬에도 해석을 제한하는 병목이 있습니다. 이것은 Disco의 일반적인 기하 학습 능력을 부정하는 결과가 아닙니다.

- [상세 실패 원인 분석과 다음 실험](docs/RESULTS_AND_NEXT_STEPS_KO.md)
- [기계 판독용 전체 결과](results/full-experiment/metrics.json), [RL CSV](results/full-experiment/rl_metrics.csv), [생성 CSV](results/full-experiment/transfer_metrics.csv)
- [설치·실험 재현·모듈별 사용법](docs/REPRODUCING.md), [Vast/L40S 운영 가이드](docs/VAST_L40S_KO.md)

## 실험 범위

| 항목 | 완료된 실험 |
| --- | --- |
| 데이터 | 절차적 물체 512개 × 12시점, 64×64 RGB; 동일 재질·고정 조명 |
| Identity split | RL train 256 / RL validation 51 / probe train 103 / probe validation 51 / test 51 |
| 비교 | 미학습 CNN, AC, AC+aux, Disco103, Disco−yz, Disco−aux |
| RL 학습 | seed 0·1·2, 학습 조건별 약 500k 유효 transitions; 초기 CNN을 seed별로 맞춤 |
| Checkpoint 선택 | 기하 점수와 독립적인 RL-validation return으로 best 선택 |
| 기하 평가 | Frozen single-image feature의 linear depth/absolute-yaw probe 및 64-unit relative-yaw MLP |
| Probe 표본 | Depth/absolute-yaw train 1,236 images; test 51 identities |
| Conditioning cache | 총 128장: adapter train 89 / validation 19 / test 20 |
| 최종 생성 | Cache test에서 4개 물체 선택; 18 encoder × 4 inputs × 3 modes + native 4개 = 220개 |
| 생성 설정 | Flow sampling 20 steps, Gaussian 32,768개, noise seed 42 하나, adapter 30 epochs |
| GPU | 저장된 실행 환경은 NVIDIA L40S |

실험 설정·완료 상태는 [pipeline.json](runs/full-experiment/pipeline.json), [study summary](runs/full-experiment/study/summary.json), [transfer plan](runs/full-experiment/transfer/plan.json), [완료 기록](runs/full-experiment/transfer/completed.json)에 보존되어 있습니다. `smoke`와 `artifact-smoke`는 연결 검증용이며 아래 연구 결과에 포함하지 않습니다.

## 인식 학습과 기하 표현 결과

아래 ±는 **3개 RL seed 간 표본 표준편차**이며 신뢰구간이 아닙니다. 정책은 선택된 best checkpoint를 deterministic argmax로 seed당 256 test episodes 평가했습니다. 기하 probe는 학습과 identity가 분리된 물체에서 평가합니다.

| 방법 | 인식 성공률 ↑ | 평균 카메라 이동 | Depth RMSE ↓ | Absolute yaw 오차 ↓ | Relative yaw 오차 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 미학습 CNN | 48.70 ± 3.52% | 0.833 | 0.4848 ± 0.0119 | 47.38° | 48.86° |
| AC | 81.64 ± 9.60% | 0.167 | 0.5374 ± 0.0568 | 66.91° | 60.02° |
| AC+aux | 90.23 ± 0.78% | 0.176 | 0.5332 ± 0.0185 | 66.37° | 55.41° |
| Disco103 | 89.32 ± 1.58% | 0.000 | 0.5631 ± 0.0205 | 65.64° | 67.25° |
| Disco−aux | 88.15 ± 0.60% | 0.000 | 0.5507 ± 0.0390 | 57.19° | 68.85° |
| Disco−yz | 64.32 ± 21.76% | 0.000 | 0.5147 ± 0.0238 | 51.23° | 50.16° |

전체 metric·seed별 수치는 [원본 집계](runs/full-experiment/study/report/summary.json)와 [RL CSV](results/full-experiment/rl_metrics.csv)에 있습니다.

Disco는 이 실행에서 기본 AC보다 높은 평균 인식 성공률과 작은 seed 변동을 보였지만 AC+aux를 넘어서는 우위는 확보하지 못했습니다. 세 Disco seed 모두 **greedy test 중 카메라를 한 번도 움직이지 않고** 약 89%를 달성했습니다. 학습 중에는 stochastic action sampling을 사용하므로, 이 결과가 훈련 내내 이동하지 않았다는 뜻은 아닙니다.

현재 기하 probe는 학습된 표현보다 미학습 CNN에서 더 좋았습니다. Probe-train 1,236장에 대한 covariance-spectrum effective rank는 미학습 CNN **28.5–36.4**, Disco **2.17–4.22**이고, Disco 최종 ReLU feature 256차원 중 **175–198차원**이 이 표본 전체에서 0입니다. 이는 task-specific compression과 부합하지만, 중간 layer나 비선형 decoder에서도 기하 정보가 없다는 증거는 아닙니다.

Depth는 **학습에 background=0을 포함하고 validation/test는 foreground만 평가**하는 목적 불일치가 있습니다. Probe-train 평균 foreground 깊이만 예측하는 상수 baseline의 test RMSE는 **0.2002**로 모든 encoder probe보다 낮습니다. 따라서 현재 depth 점수를 기하 정보의 부재로 단정하지 않습니다.

![Depth·adapter·feature의 사후 진단](runs/full-experiment/audit/audit_summary.png)

## Frozen TripoSplat 전이 결과

RL encoder와 pretrained generator를 고정한 뒤, RL feature에서 TripoSplat의 native image-conditioning을 예측하는 adapter만 학습했습니다. Generator 전체를 fine-tuning한 실험이 아닙니다.

**아래는 전부 UNREGISTERED orbit diagnostics입니다.** 생성물의 내부 좌표를 benchmark 좌표에 정렬하지 않았고, CPU renderer는 공식 rasterizer의 근사 구현입니다. 입력 시점을 제외한 11개 view를 물체별로 평균하고, 물체를 RL seed별로 평균합니다. 실제 기하 품질의 순위로 해석하지 않습니다.

| 조건 | 전체 PSNR ↑ | Foreground PSNR ↑ | Silhouette IoU ↑ |
| --- | ---: | ---: | ---: |
| Native RGB | 10.132 | 9.191 | 0.3971 |
| Zero conditioning | 11.324 | 9.249 | 0.4928 |
| 미학습 CNN → adapter | 11.243 | 8.852 | 0.4833 |
| AC → adapter | 10.922 | 8.674 | 0.4896 |
| AC+aux → adapter | 10.820 | 8.454 | 0.4776 |
| Disco → adapter | 11.206 | 8.543 | 0.4739 |
| Disco shuffled | 10.782 | 8.149 | 0.4291 |

전체 19개 조건의 평균·표준편차·seed별 값은 [생성 CSV](results/full-experiment/transfer_metrics.csv), 물체별 paired 값은 [원본 결과](runs/full-experiment/transfer/report/per_image.csv)에 있습니다. Native는 공통 baseline이며 zero의 반복도 독립적인 RL-seed 실험으로 세지 않습니다. 물체는 4개, noise seed는 하나뿐입니다.

Disco learned가 shuffled보다 높아도 미학습 CNN에서도 같은 방향이 나타나므로 RL의 추가 이득은 확인되지 않았습니다. 정성 예시에서 native는 L자 형태를 어느 정도 보존하지만 learned 조건들은 비슷한 막대로 보입니다. 정답과 다른 유색 형상을 만드는 zero가 높은 점수를 받는 사례도 있어, 좌표와 평가 검증이 필요합니다.

![Scene 49, seed 0: 입력 시점 외의 비교. 좌표 정렬 전 진단](runs/full-experiment/transfer/report/qualitative_scene49_seed0.png)

Adapter 자체에서도 명확한 fitting 병목이 있습니다. Seed0 Disco의 validation MSE는 **0.53847**, train teacher를 이미지 축으로 평균한 token/channel별 상수 tensor를 출력하는 baseline은 **0.42110**입니다. Train full-token MSE도 **0.52735 > 0.40934**입니다. 모델의 per-token bias는 이 상수 해를 표현할 수 있으므로, 실패를 encoder의 정보 부족만으로 설명할 수 없습니다. 18개 adapter 모두 마지막 30 epoch에서 best를 기록해 충분한 수렴도 확인되지 않았습니다. 세부 진단은 seed0의 선택된 3개 방법과 validation 19장 범위입니다.

## 실패 원인의 해석

| 항목 | 확인된 증거 | 현재 가능한 해석 |
| --- | --- | --- |
| Task와 가설의 연결 | 즉시 답변 허용, 무작위 negative, greedy Disco 이동 0 | 초기 RGB 쌍만으로 높은 정확도를 얻어 다중 시점 정보 수집의 필요성이 약함 |
| 관측 memory | 현재/reference RGB만 처리; history memory 없음 | 여러 시점의 누적 증거를 통합하는 정책을 학습할 구조가 없음 |
| 표현 압축 | 최종 feature의 effective rank 감소와 표본 내 zero dimension 증가 | Task 보상이 기하 정보 보존보다 인식에 유용한 압축을 유도했을 가능성 |
| Depth 측정 | Dense fitting과 foreground-only scoring 불일치; 상수가 더 좋음 | 표현 차이와 probe 목적 불일치가 섞여 있음 |
| Adapter fitting | 표현 가능한 상수 baseline보다 train/validation 모두 나쁨 | Conditioning 평균과 입력별 변화를 회수하는 최적화가 미완성 |
| 생성 평가 | 좌표 정렬 없음, 근사 renderer, 4 objects·1 noise seed | 수치 차이를 기하 품질이나 통계적으로 확립된 우위로 연결하기 어려움 |

앞의 구조·수치는 관찰된 사실입니다. Reward shortcut, 표현 압축, adapter 최적화가 각각 실패에 기여한 양은 개입 실험으로 분리해야 합니다. 상세 분석에는 loss ablation의 checkpoint 차이, token sampling budget, loss의 head별 가중치, shuffled control의 분포 차이도 기록했습니다.

## 다음 실험

1. 기존 frozen feature로 foreground-aware depth probe와 상수·RGB·silhouette baseline을 검증합니다.
2. Adapter를 teacher 평균에서 residual을 예측하도록 구성하고, feature 표준화·간단한 linear/PCA baseline·충분한 수렴을 확인합니다. Validation에서 평균 predictor를 이기는 것을 통과 기준으로 둡니다.
3. 초기 view는 모호하고 다른 방향에서 구별되는 hard pair와 temporal memory를 도입합니다. No-move·고정 탐색·learned 탐색을 동일 episode에서 비교합니다.
4. 같은 backbone과 memory에서 Disco와 AC+aux의 layer별 기하 probe·rank·인식 성능을 추적합니다.
5. Validation에서 camera/좌표/renderer를 검증한 뒤, 미관찰 test 물체와 생성 noise seed를 확대합니다.

현재 결과에 가장 잘 맞는 후속 질문은 **“다중 시점의 필요성이 task reward로 학습한 표현의 기하 정보 보존에 어떤 영향을 주는가?”**입니다. 각 단계의 검증 방법과 통과 기준은 [상세 분석](docs/RESULTS_AND_NEXT_STEPS_KO.md)에 있습니다.

## 실행과 결과 재집계

설치 및 전체 파이프라인의 상세 옵션은 [재현 가이드](docs/REPRODUCING.md)를 참고합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# GPU 작업과 다운로드 없이 파이프라인 확인
bash scripts/run_full_experiment.sh --dry-run

# 보존된 실험 결과를 재집계·교차 검증 (Python 표준 라이브러리만 사용)
python3 scripts/summarize_results.py \
  --run runs/full-experiment --output results/full-experiment
```

[metrics.json](results/full-experiment/metrics.json)은 원본 summary와 진단 수치를 담고, [provenance.json](results/full-experiment/provenance.json)은 입력 파일의 SHA-256과 집계 검증 내역을 기록합니다. 이 재집계는 기존 산출물의 수치 검증이며 RL·adapter 재학습, pretrained 생성 재실행, renderer parity 검증을 수행하지 않습니다.

RL 학습에는 RGB·정답 보상·이동 비용만 사용하며 depth/pose label이나 generative-teacher gradient는 들어가지 않습니다. 공개 Disco103의 policy/KL/value/next-policy loss는 유지합니다. `no_yz_loss`와 `no_aux_policy`는 frozen rule의 perturbation이며 새로 발견한 알고리즘이 아닙니다. 기존 CPU smoke/integration 검증은 구현 연결을 확인하며 기하 emergence 자체를 검증하지 않습니다.

## 원본과 출처

- [DiscoRL 공식 구현](https://github.com/google-deepmind/disco_rl), [논문](https://www.nature.com/articles/s41586-025-09761-x)
- [TripoSplat 공식 inference](https://github.com/VAST-AI-Research/TripoSplat), [논문](https://arxiv.org/abs/2605.16355), [training code](https://github.com/runjie-yan/TripoSplat-Training)
- [고정된 upstream commit·체크섬·라이선스](third_party/PROVENANCE.json)
- [기존 audit 원본](runs/full-experiment/audit/diagnostics.json), [학습 곡선](runs/full-experiment/study/report/learning_curves.png), [snapshot probe](runs/full-experiment/study/report/snapshot_probes.csv)

결과 정리와 분석 갱신: 2026-10-28. 이 문서 갱신에서 새 RL 학습이나 3D 생성은 실행하지 않았습니다.
