# DiscoGeo: L40S / Vast.ai 실행 안내

확인일: 2026-10-06. 명령은 저장소 루트에서 실행한다.

## 서버와 설치 파일

**L40S 1장으로 시작할 수 있다.** 공식 사양은 48 GB VRAM, FP16/BF16 지원이다.
현재 연구는 작은 CNN의 RL 학습과 동결된 TripoSplat 추론을 순차 실행한다.
전체 TripoSplat을 학습하는 구성은 아니다. 48 GB는 시작하기 좋은 용량이지만,
전체 사전학습 모델 추론은 아직 L40S에서 실측하지 않았다.
첫 `generation.json`에 GPU 이름과 PyTorch peak allocated/reserved 메모리가 기록된다.
프로세스 전체 VRAM 수치는 아니므로 `nvidia-smi`도 확인한다.

| 항목 | 시작 설정 |
| --- | --- |
| GPU | L40S 48 GB × 1 |
| RAM | 32 GB 이상, 가능하면 64 GB |
| CPU | 8 vCPU 정도 |
| writable SSD | **100 GB 권장**, 관리하면서 쓰면 50–80 GB도 가능 |
| 이미지 | Linux x86_64, Ubuntu 22.04/24.04, Python 3.12, CUDA 준비된 이미지 |
| 호스트 드라이버 | CUDA 12.6 이상을 지원하는 호스트; 설치 후 GPU 검사 |
| 인스턴스 | 첫 실험은 on-demand 권장; interruptible은 중단을 감수할 때 |

컨테이너에 NVIDIA 호스트 드라이버를 설치하지 않는다. CUDA 라이브러리는 pip wheel로 설치한다.
Conda는 필수가 아니며 Python 3.12 + venv면 된다.

- `requirements.txt`: RL, 생성 확장, matplotlib, pytest/ruff까지 전체 Python 의존성.
- `requirements-vast.txt`: Linux GPU 프로필. JAX 0.4.38 CUDA 12, PyTorch 2.6.0 /
  torchvision 0.21.0 CUDA 12.6 wheel 지정.
- `requirements-system.txt`: git/tmux/rsync, Python venv 등 OS 조건. **pip 입력용이 아니다.**
- `environment.yml`: Conda를 이미 쓰는 경우의 선택지, Python 3.12 지정.
- `scripts/setup_vast.sh`: venv 설치와 JAX/PyTorch GPU 확인.

서버 프로필은 설치 후보를 명시한 것이다. 로컬 검증은 macOS Python 3.12 CPU에서
진행했으며 Linux CUDA 조합은 실제 서버에서 GPU 검사를 통과해야 한다.
최신 CUDA 13 wheel을 자동 선택하는 일을 피하려고 CUDA 12 wheel을 지정했다.
Python, tmux, Conda 자체와 호스트 드라이버는 pip 패키지로 넣을 수 없어서 주석과
OS 목록/환경 파일로 분리했다.

## 업로드와 설치

처음 실행할 때의 순서는 **SSH 키 등록 → 서버 대여 → 서버 점검/전송 도구 준비 →
코드 업로드 → tmux → 환경 설치 → 검사 → 전체 실험**이다.
Vast 계정에 SSH 공개키를 등록한 뒤 Python 3.12가 포함된 CUDA 이미지로 서버를 만든다.
Vast UI의 SSH 연결 정보를 사용한다. 아래 `PORT`, `HOST`는 실제 값으로 바꾼다.
사용자명이 root가 아니면 UI에 나온 사용자명으로 바꾼다.

맥 터미널에서 먼저 연결한다:

```bash
ssh -p PORT root@HOST
```

서버 터미널에서 GPU/Python을 확인하고 업로드에 필요한 도구와 경로를 준비한다:

```bash
nvidia-smi
python3.12 --version
apt-get update
apt-get install -y git tmux rsync ca-certificates
mkdir -p /workspace/disco-geo
```

위 apt 명령은 root 컨테이너 기준이다. 일반 사용자면 관리자 권한으로 실행한다.
Python 3.12가 없는 이미지에서는 먼저 해당 런타임을 준비해야 한다.
이제 **맥의 다른 터미널**에서 코드만 올린다. 서버에서는 데이터와 가중치를 자동 준비한다.
예시는 기존 원격 파일을 지우지 않는다.

```bash
rsync -av --exclude='.venv*' --exclude='.git' --exclude='runs/' \
  --exclude='datasets/' --exclude='checkpoints/' --exclude='__pycache__/' \
  -e 'ssh -p PORT' \
  /Users/jongbinkim/Documents/ChatGPT/disco-geo/ root@HOST:/workspace/disco-geo/
```

서버에서는 tmux 세션을 준비한다. Vast SSH가 이미 tmux 안에서 시작할 수 있으므로,
중첩 세션을 만들지 않고 discogeo 세션으로 전환한다:

```bash
tmux has-session -t discogeo 2>/dev/null || tmux new-session -d -s discogeo
if [ -n "${TMUX:-}" ]; then
  tmux switch-client -t discogeo
else
  tmux attach -t discogeo
fi
```

이후 **discogeo 세션 안의 서버 터미널**에서 설치한다:

```bash
cd /workspace/disco-geo
bash scripts/setup_vast.sh
source .venv.cuda/bin/activate
python -m pip check
python -m pytest -q
python -m disco_geo --help
```

전체 실행 계획 확인 후 실제 실험을 시작한다:

```bash
bash /workspace/disco-geo/scripts/run_full_experiment.sh --dry-run
bash /workspace/disco-geo/scripts/run_full_experiment.sh
```

`dry-run`은 실행할 명령만 출력한다. 기본 전체 실험은 데이터 준비/가중치 다운로드/
조건 캐시 생성/ablation/최종 추론/비교를 자동 실행하므로 이 단계들을 별도로 실행할 필요는 없다.

시스템 도구가 설치되어 있고 apt 권한이 없으면 `--skip-system`을 사용한다.
Python 3.12 venv 모듈이 없으면 이미지에 맞는 `python3.12-venv`를 설치한다.
Conda를 이미 사용한다면 venv 대신:

```bash
conda env create -f environment.yml
conda activate discogeo
sudo apt-get update
sudo apt-get install -y git tmux rsync ca-certificates
```

## tmux와 Inactive

**SSH 종료/노트북 종료는 괜찮지만, Vast Stop/Inactive에서는 학습이 멈춘다.**
공식 문서에서 Inactive는 중지되었지만 디스크 데이터가 보존된 상태다.
tmux는 실행 중인 서버에서 연결이 끊겨도 세션을 유지한다. 중지된 컨테이너에서
GPU 작업을 계속하거나 재시작 후 프로세스를 복원하는 기능은 아니다.

```bash
tmux new -s discogeo
cd /workspace/disco-geo
source .venv.cuda/bin/activate
mkdir -p runs/logs
set -o pipefail
# 아래 연구 명령을 이 세션 안에서 실행한다.
```

분리: `Ctrl+b` 다음 `d`. 이후 SSH 창을 닫아도 된다. 다시 연결한 뒤:

```bash
tmux attach -t discogeo
```

학습 중에는 Running으로 유지하고 GPU 사용 요금을 지불한다. Stop 후에도 저장 비용이
남을 수 있고 재시작 때 GPU가 즉시 배정되지 않을 수 있다. Destroy는 디스크도 삭제하므로
먼저 로컬에 결과를 복사한다. 재시작은 `--resume`으로 `latest.pkl`부터 새로 실행한다.
마지막 원자적 체크포인트 이후 작업은 다시 수행한다. 첫 체크포인트 이전에 중단되면
새 output 디렉터리로 시작한다.

## 예상 시간을 실제 서버에서 계산하기

데이터, 가중치, native 조건 캐시는 한 번 만들어 재사용한다.

```bash
python -m disco_geo prepare --output datasets/objects --scenes 512 --parts 8
python -m disco_geo download-triposplat --output checkpoints/triposplat
python -m disco_geo cache-conditioning --dataset datasets/objects \
  --weights checkpoints/triposplat --output datasets/tripo_cache --max-items 128

python -m disco_geo benchmark-rl --config configs/objects.json \
  --method disco --output runs/timing/disco.json
python -m disco_geo benchmark-rl --config configs/objects.json \
  --method actor_critic --output runs/timing/actor_critic.json
python -m disco_geo benchmark-rl --config configs/objects.json \
  --method actor_critic_aux --output runs/timing/actor_critic_aux.json
```

출력은 실제 장치, 최초 컴파일/워밍업 시간, 유효 transition/s, 평가 시간과
500k transition의 조건부 소요 시간이다. `steps`는 유효 환경 transition 수,
snapshot의 `iter`는 learner update 수다. 정책에 따라 reset 비율과 처리량이 바뀐다.

**L40S 전체 시간을 아직 실측하지 않았으므로 특정 시간을 보장할 수 없다.**
3 seed + ablation이면 5 변형 × 3 seed × 500k = **7.5M transition**이다.
untrained 3회는 RL 업데이트를 하지 않는다. 다음은 L40S 성능 수치가 아닌 산술 예시다.

| 실제 평균 처리량이 이렇게 측정됐다면 | RL 업데이트 연산 시간 |
| --- | --- |
| 100 transition/s | 약 20.8시간 |
| 500 transition/s | 약 4.2시간 |
| 1,000 transition/s | 약 2.1시간 |

총시간은 `RL 학습·평가 + 스냅샷별 probe/adapter 학습 + 3D 생성 + orbit 렌더링 +
최초 설치/다운로드/조건 캐시`다. 학습 run마다 5 스냅샷, untrained마다 1개,
객체 2개, learned 모드 하나라면 **78 adapter + 156 생성**이다.
adapter당 2분, 생성당 1분으로 *측정되었다는 가정*에서는 각각 2.6시간이 추가된다.
RL이 평균 500 transition/s라면 이 세 단계만 약 9.4시간이고 나머지 비용이 더해진다.
실측이 다르면 추정도 바뀐다. 최초에는 작은 1 seed pilot으로 측정한다.
모드 4개를 매 스냅샷 생성하면 생성 비용은 약 4배다.

pilot에서 얻은 adapter `result.json`, 생성 `generation.json`으로 계산할 수 있다.
`SNAPSHOT`과 `SCENE`은 실제 `iter0001000`, `sceneN_viewV` 폴더 이름으로 바꾼다.

```bash
python scripts/estimate_runtime.py --profiles runs/timing/*.json --ablations \
  --snapshots-per-trained-run 5 --items 2 --modes 1 \
  --adapter-result runs/study/disco-none/seed0/snapshots/SNAPSHOT/adapter/result.json \
  --generation-metadata runs/study/disco-none/seed0/snapshots/SNAPSHOT/generation/SCENE/learned/generation.json
```

이것도 측정된 일부 단계의 추정 합이다. 설치/캐시/probe/CPU 렌더링과 각 RL 프로세스
최초 컴파일은 별도다. `completed.json`, `study_timing.json`, cache manifest,
adapter result, generation metadata에 실제 경과 시간이 남는다.
생성에서는 모델 로드/내보내기를 포함한 프로세스 내 시간과 sampling 시간을 구분한다.

## Ablation, 주기적 3D, 그래프

전체 과정을 한 번에 실행하려면 설치 후 tmux 안에서:

```bash
bash scripts/run_full_experiment.sh
# 중단 후 동일한 설정으로 재개
bash scripts/run_full_experiment.sh --resume
# 실행 계획만 확인
bash scripts/run_full_experiment.sh --dry-run
```

이 파일은 데이터/가중치/조건 캐시 준비, 6개 변형 × 3 seed 학습과 probe,
주기적 검증 3D, 각 변형의 **best 체크포인트**에 대한 최종 adapter 학습,
동일한 test 객체 4개로 native/learned/zero/shuffled 추론·평가·비교 그래프까지 실행한다.
최종 추론은 기본 220회이며 주기적 검증 생성이 추가된다.
결과는 `runs/full-experiment/study/report`, `transfer/report`, 로그는 `logs/pipeline.log`에 저장한다.
`transfer/report`에는 이미지별/시점별/seed별 수치와 동일 시점의 정성 비교 패널도 포함된다.
최종 수치의 표준편차는 이미지와 시점을 중복 seed로 취급하지 않고 RL seed별 평균으로 계산한다.

`--snapshot-every 0`이면 중간 3D를 생략하고 최종 비교까지 실행한다.
`--seeds 0 --steps 50000 --test-items 2`처럼 pilot 규모를 바꿀 수 있다.
`--output runs/pilot`으로 전체 실험과 다른 폴더를 지정한다.
설치까지 함께 하려면 `--setup`, Conda 등 활성 환경을 지정하려면 `--python /path/to/python`을 사용한다.
`--help`에 전체 옵션이 있다. 재개할 때는 기존 옵션을 모두 동일하게 지정한다.

tmux에서 아래를 실행한다. 1,000 learner update마다 CNN을 저장하며 초기/최종도 저장한다.
각 run의 RL 학습 후 별도 프로세스가 해당 iter CNN으로 probe와 3D를 만든다.
그 iter의 결과이지만 학습과 동시에 3D 모델을 돌리지는 않는다.
스냅샷마다 같은 seed/예산으로 새 adapter를 학습한다. 최종 adapter를 이전 스냅샷에
그대로 붙이지 않는다. RL 손실/보상에는 생성 MSE를 추가하지 않는다.

```bash
python scripts/run_study.py --config configs/objects.json --output runs/study \
  --steps 500000 --seeds 0 1 2 --ablations --snapshot-every 1000 \
  --snapshot-cache datasets/tripo_cache --snapshot-weights checkpoints/triposplat \
  --snapshot-items 2 --snapshot-epochs 30 2>&1 | tee -a runs/logs/study.log
```

재시작 후 동일 명령에 `--resume`을 더한다. 완료된 downstream 단계는 재사용하며,
중단된 단계는 새 임시 폴더에서 실행 후 완료 시 공개한다. 다른 실험은 새 output을 사용한다.
`--resume`은 모델/데이터/학습 설정을 확인하며 목표 steps는 늘릴 수 있다.

특정 iter만 선택하는 계획 예시:

```bash
python scripts/run_snapshots.py --run runs/study/disco-none/seed0 \
  --cache datasets/tripo_cache --weights checkpoints/triposplat --items 2 \
  --iterations 0 1000 2000 --modes learned zero shuffled native --plan-only
```

`--plan-only`를 빼면 실행한다. 이미 다른 설정으로 처리한 iter는 덮어쓰지 않으므로,
처음부터 원하는 모드를 정해 같은 설정으로 `--resume`하거나 별도 실험 run을 사용한다.
수치만 먼저 보려면 `run_study.py`의 cache/weights 대신 `--snapshot-probes`, 또는
`run_snapshots.py --run ... --probes-only`를 사용한다.

```bash
python -m disco_geo report --study runs/study
```

`report/`에는 benchmark/learning curve/snapshot probe CSV, seed별 수치·평균·표준편차 JSON,
PNG/PDF 그래프, 깊이 정답·예측 패널이 남는다. 한 seed의 표준편차는 정의하지 않는다.
주기적 probe/3D는 고정 **probe_val** identity를 쓰고 최종 test와 분리한다.
현재 3D orbit 수치는 좌표 정합 전 진단이고 그래프에도 **UNREGISTERED**라고 표시한다.
생성물만으로 기하 능력 emergence를 주장하지 않는다.

## 저장 정책과 용량

| 보존 대상 | 목적 |
| --- | --- |
| config/provenance/seed/split/hash, JSONL/CSV/JSON | 재현 및 ablation 수치 해석 |
| best.pkl / latest.pkl | 검증 선택 모델 / 학습 복원, 각각 계속 덮어쓰기 |
| encoder.pkl + iteration.json | 그 iter의 CNN, optimizer 없이 저장 |
| features, probe predictions/targets | 재평가 및 깊이 오류 확인 |
| adapter best.pt, epoch loss, result.json | CNN→3D 조건 변환 재현 |
| object.ply + gaussians.npz, camera/seed/메타데이터 | 보기 및 재렌더링/평가 |
| 소수 orbit PNG, 수치, PNG/PDF 그래프 | 정성·정량 해석 |

매 epoch optimizer/전체 diffusion latent/중복 SPLAT/전체 영상은 기본으로 쌓지 않는다.
PLY와 NPZ는 viewer 변환 좌표와 내부 평가 좌표 역할이 달라 둘 다 남긴다.
추가 형식이 필요하면 `generate --exports ply npz splat`을 선택한다.
기록을 자동 삭제하지 않는다. pip 캐시는 설치 시 `--no-cache-dir`로 줄인다.

512 객체 × 12 view × 64px 데이터(RGB/깊이/법선/마스크)는 원소 크기로 약 **0.35 GB**,
가중치 5개는 약 **3.78 GB**, 조건 128개는 fp16 원소 기준 약 **1.48 GB**다.
feature array는 CNN당 약 **6 MiB**, 기본 전체 RL 체크포인트는 로컬 측정 약 **21 MiB**다.
CNN 스냅샷은 더 작다. 32k Gaussian과 소수 객체, 드문 스냅샷이면 300 GB가 필수일 이유는 없다.
CUDA 환경/설치 임시 공간/실험 수를 고려해 100 GB에서 시작하고 `du -sh`로 확인한다.
스냅샷 간격을 너무 줄이면 adapter 재학습 시간과 보존 수가 크게 늘어난다.

백업 예시(로컬에서):

```bash
rsync -av -e 'ssh -p PORT' root@HOST:/workspace/disco-geo/runs/ \
  /Users/jongbinkim/Documents/ChatGPT/disco-geo/runs/
```

## 확인한 공식 자료

- [NVIDIA L40S 사양](https://www.nvidia.com/en-us/data-center/l40s/)
- [Vast 상태·Stop·재시작](https://docs.vast.ai/guides/instances/manage-instances)
- [Vast Docker 환경](https://docs.vast.ai/guides/instances/docker-environment)
- [Vast SSH 연결과 기본 tmux 세션](https://docs.vast.ai/guides/instances/connect/ssh)
- [tmux detach/attach](https://github.com/tmux/tmux/wiki/Getting-Started)
- [PyTorch 공식 CUDA wheel 조합](https://pytorch.org/get-started/previous-versions/)
- [JAX GPU 설치 조건](https://docs.jax.dev/en/latest/installation.html)
- [TripoSplat 공식 구현](https://github.com/VAST-AI-Research/TripoSplat)
