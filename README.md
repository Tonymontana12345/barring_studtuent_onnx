# Bearing KD ONNX

**한국어** | [English](README.en.md)

Paderborn 진동 데이터의 정상/고장을 분류하는 실험 프로젝트입니다. MOMENT Base의 LoRA와 분류 헤드를 학습한 Teacher를 이용해 MOMENT Small의 **분류 헤드만** 증류하고, Raspberry Pi CPU 추론용 ONNX로 변환합니다.

> **현재 상태 요약 (2026-09-25)**
> - 배포 경로(ONNX 변환 → Raspberry Pi 로드 → Mac과 수치 일치 → 실시간 속도)는 **검증 완료**입니다.
> - 모델 성능은 **아직 배포 수준이 아닙니다.** 학습에 없던 베어링으로 평가하면 Student 64.6%이고, KB24 고장을 **0%** 검출합니다.
> - 같은 베어링의 다른 파일로 만든 validation(98~100%)은 실제 성능을 크게 부풀립니다. 자세한 내용은 [v3 실험](#v3-실험-베어링-단위-holdout-평가)을 참고하세요.

## 구조

```text
src/bearing_kd/
  preprocessing.py  # 모든 베어링의 공통 전처리·라벨
  data.py           # 파일 분할, Dataset, 청크, 캐시 무결성
  models.py         # MOMENT/LoRA, 엄격한 체크포인트 로딩
  training.py       # Teacher 학습, 예측 캐시, Student KD, 평가
  export.py         # ONNX 변환 + PyTorch 수치 비교
  inference.py      # PyTorch 없이 윈도우 단위 ONNX 추론
  cli.py            # bearing-kd 명령
scripts/analysis/   # 파형·주파수·임베딩 분석
scripts/verify_pipeline.py # 실제 데이터/모델 통합 검증
scripts/download_models.py # 사전학습 모델 준비
scripts/pi_verify.py       # Raspberry Pi ONNX 수치 검증 (numpy + onnxruntime만 필요)
tests/             # 다운로드 없이 실행하는 자동 회귀 테스트
data/splits/       # 재현용 파일 목록 (Git 포함)
docs/              # 구조·이전 코드·검증 기록
```

원본/가공 데이터, 모델 가중치, outputs, 가상환경은 `.gitignore`로 제외합니다. Git 초기화나 원격 업로드는 수행하지 않았습니다.

## 설치

Python **3.11**과 `uv`를 사용합니다. MOMENT 0.1.4의 오래된 numpy/transformers/huggingface-hub 고정 의존성이 현재 PEFT와 충돌하므로, 검증한 버전을 `constraints-tested.txt`에 명시하고 uv override로 설치합니다. 일반 `pip install -r requirements.txt`만으로는 이 충돌을 해결할 수 없습니다.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-lock.txt --override constraints-tested.txt
bearing-kd --help
```

현재 작업 폴더에서 검증한 기존 환경은 `.venv-uv`입니다.

```bash
source .venv-uv/bin/activate
bearing-kd evaluate-student
```

ONNX 추론만 필요한 Raspberry Pi 등에서는 PyTorch/MOMENT 없이 설치할 수 있습니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
```

> **Raspberry Pi 주의:** 검증한 Pi 환경은 Python **3.13** 가상환경(`~/vib-onnx/.venv`)입니다. 그런데 `pyproject.toml`은 `requires-python = ">=3.11,<3.12"`라서 Pi에서 `pip install -e .`(`bearing-kd infer`)가 설치를 거부합니다. 지금은 `scripts/pi_verify.py`처럼 numpy + onnxruntime만 쓰는 독립 스크립트를 사용하세요. 가상환경을 활성화하지 않고 실행하면 `No module named 'onnxruntime'` 에러가 납니다. `.venv/bin/python`으로 직접 실행하면 활성화 없이도 됩니다.

버전 기록은 macOS ARM64 기준이며 다른 플랫폼의 wheel 제공 여부는 다를 수 있습니다. MOMENT의 원래 메타데이터 때문에 `pip check`는 override한 세 패키지를 충돌로 표시할 수 있습니다. 설치 및 실행 검증 범위는 [검증 기록](docs/verification.md)을 확인하세요.

## 데이터와 모델 준비

[공식 Paderborn Bearing DataCenter](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter)에서 데이터를 받고 압축을 풀어 `data/raw/paderborn/K001/*.mat`처럼 배치합니다. 기존 실험은 K001(정상), KA03·KA04(고장)를 사용합니다. KB·KI도 고장 1로 지원하지만 데이터 추가 후에는 분할·캐시·학습을 다시 수행해야 합니다.

`.rar`는 macOS 기본 `bsdtar`로 풀 수 있습니다(`bsdtar -xf KA05.rar`). 베어링 하나가 풀면 약 670 MB입니다. 현재 로컬 상태는 다음과 같습니다.

| 상태 | 베어링 |
|---|---|
| 압축 해제 완료 (80파일) | K001–K006, KA03, KA04, KA05, KB24, KB27, KI03 |
| **부분 해제 (13/80) — 정리 필요** | **KA06** |
| `.rar`만 있음 | KA07–KA09, KA15, KA16, KA22, KA30, KI01, KI04, KI05, KI07, KI08, KI14, KI16–KI18, KI21 |

사전학습 모델은 [MOMENT Base](https://huggingface.co/AutonLab/MOMENT-1-base), [MOMENT Small](https://huggingface.co/AutonLab/MOMENT-1-small)입니다.

```bash
python scripts/download_models.py
```

스크립트는 `config.json`, `model.safetensors`, 모델 카드를 받습니다. `.bin`과 `.safetensors`를 둘 다 받을 필요는 없습니다. 다운로드 커밋은 `models/download_manifest.json`에 기록되며 `--base-revision`, `--small-revision`으로 특정 커밋을 지정할 수 있습니다. 기존 모델 폴더를 보존하려면 별도의 `--output-dir`을 사용하세요.

**학습된 체크포인트는 Git에 포함되지 않습니다.** 새 clone에서 평가하려면 별도로 체크포인트를 받거나 아래 학습 과정을 실행해야 합니다. 이 저장소에 공개 체크포인트 다운로드 주소는 아직 지정되어 있지 않습니다.

## 전처리 계약

| 항목 | 값 |
|---|---|
| 채널 | `vibration_1`, 단일 축 |
| 원본 샘플링 | 64,000 Hz |
| 저역통과 | 1,000 Hz, Butterworth 8차, `sosfiltfilt` |
| 리샘플링 | 4,000 Hz |
| 윈도우 / 이동 간격 | 4,000 / 2,000 샘플 |
| MOMENT 청크 | 512 샘플 × 8개 |
| 패딩 | 마지막 96개는 0, 해당 마스크는 0 |
| 라벨 | K+숫자 3자리: 정상 0 / KA·KB·KI: 고장 1 |
| 윈도우 판정 | 8개 청크의 **logits 평균** 후 argmax |

모델 내부에 RevIN이 있으므로 별도 윈도우 정규화를 하지 않습니다. 양방향 필터는 녹음 구간을 처리하는 방식이며 실시간 스트리밍용 상태 유지 필터는 포함하지 않습니다.

### 4 kHz를 선택한 이유와 대가

64 kHz는 **샘플링 레이트**(1초에 값을 기록하는 횟수)이지 신호 주파수가 아닙니다. 64 kHz로 기록하면 0~32 kHz까지 담깁니다. 4 kHz로 낮추면 담을 수 있는 최대 주파수는 절반인 **2 kHz**입니다(나이퀴스트). 그보다 빠른 진동은 샘플 사이에서 흔들리고 지나가서 복원할 수 없고, 필터 없이 낮추면 가짜 저주파로 둔갑합니다(에일리어싱). 그래서 1 kHz 저역통과로 먼저 지운 뒤 리샘플링합니다.

MOMENT 입력은 512 샘플로 고정이므로 샘플링 레이트가 곧 "보는 시간"과 "보는 대역"을 정합니다.

| 샘플링 | 512 샘플 = | 볼 수 있는 대역 | Pi 비용 (윈도우당 청크) |
|---|---|---|---|
| 64 kHz | 8 ms | 0~32 kHz | 125 |
| 16 kHz | 32 ms | 0~8 kHz | 32 |
| **4 kHz (현재)** | **128 ms** | **0~2 kHz** | **8** |

4 kHz는 **Raspberry Pi 추론 속도**를 고려한 선택입니다. 그 대가로 결함 정보가 많은 고주파 대역을 버립니다. 현장 센서가 수 kHz 이하일 수 있어서, 현재는 **고주파는 포기하고 0~1 kHz 안에서 일반화를 높이는 방향**(베어링 단위 validation, 학습 베어링 추가)으로 진행합니다.

## 새 실험 실행

생성 명령은 출력 위치를 명시해야 합니다. 기존 결과와 섞이지 않게 새로운 경로를 사용하세요. 아래 실행은 학습 시간이 걸립니다.

```bash
# 1. 공통 전처리
bearing-kd preprocess --bearings K001 KA03 KA04 --output-dir data/processed/windows_v2
# 2. 녹음 파일 단위 70/15/15 분할
bearing-kd split --processed-dir data/processed/windows_v2 --output data/splits/file_split_v2.json
# 3. Teacher LoRA + 분류 헤드 학습
bearing-kd train-teacher --split data/splits/file_split_v2.json --output-dir outputs/runs/v2/teacher
# 4. Teacher 예측 캐시
bearing-kd cache-teacher --split data/splits/file_split_v2.json --checkpoint outputs/runs/v2/teacher/teacher_lora_best.pt --output-dir outputs/runs/v2/cache
# 5. Student 분류 헤드 증류
bearing-kd train-student --split data/splits/file_split_v2.json --cache-dir outputs/runs/v2/cache --output-dir outputs/runs/v2/student
# 6. Test 평가
bearing-kd evaluate-student --split data/splits/file_split_v2.json --checkpoint outputs/runs/v2/student/student_kd_best.pt
# 7. ONNX 변환 + 수치 동등성 검사
bearing-kd export --checkpoint outputs/runs/v2/student/student_kd_best.pt --output outputs/runs/v2/onnx/student.onnx
```

**베어링 단위 holdout 분할**은 `--test-bearings`로 지정합니다. 지정한 베어링의 파일은 전부 test로 가고, 나머지 베어링은 파일 단위로 train/validation 85/15로 나뉩니다. test에 정상과 고장 베어링이 모두 없으면 거부합니다. 고장 베어링만 있으면 "항상 고장"이라고 답하는 모델도 100%가 되기 때문입니다.

```bash
bearing-kd split --processed-dir data/processed/windows_v3 --test-bearings K004 KB24 KB27 --output data/splits/bearing_holdout_v3.json
```

기본값은 CPU, 20 epoch, batch 16, learning rate 1e-4, seed 42입니다. `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--device cuda|mps`로 변경할 수 있습니다.

**Apple Silicon:** CUDA는 NVIDIA 전용이라 Mac에서는 `--device mps`(Metal)를 씁니다. 지원하지 않는 연산은 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정하면 CPU로 넘어갑니다. 측정 결과 Teacher 학습(batch 16) 속도는 CPU 6.4초, MPS 3.1초였습니다. 같은 seed라도 CPU와 결과가 비트 단위로 같지는 않습니다.

**학습 시간:** LoRA 버그를 고친 뒤에는 역전파가 인코더 전체를 통과하고, gradient checkpointing 때문에 순전파를 한 번 더 계산합니다. 그래서 이전보다 약 3~4배 느립니다. v3(약 2,860개 윈도우) Teacher는 MPS에서 epoch당 약 10.7분이었습니다. 이전의 "20 epoch 1시간"은 LoRA가 실제로 학습되지 않아서 빨랐던 것입니다.

**체크포인트 주의:** 현재 코드는 validation 정확도가 **같거나 높으면** `*_best.pt`를 덮어씁니다. validation이 100%에 붙으면 가장 늦은 100% epoch만 남습니다. v3에서는 epoch 4 모델이 9와 10에 덮어쓰였습니다. epoch별 저장은 아직 구현하지 않았습니다.

Student KD는 temperature 2, 정답 손실 비중 0.5이며 backbone을 고정합니다. 설정·데이터 해시·epoch별 결과는 학습 폴더에 기록합니다. 같은 seed라도 하드웨어와 라이브러리에 따라 결과가 달라질 수 있습니다.

경로는 프로젝트 루트를 기준으로 해석합니다. 다른 데이터 위치나 일반 설치에서는 `bearing-kd --root /path/to/workspace ...`를 사용하세요. 공통 옵션 `--root`, `--threads`는 하위 명령 **앞에** 씁니다.

## 기존 결과 평가와 추론

기존 로컬 데이터·체크포인트가 있을 때:

```bash
bearing-kd evaluate-student
bearing-kd evaluate-teacher
bearing-kd evaluate-teacher --processed-dir data/processed/windows_1khz/KB24
bearing-kd infer --input data/processed/windows_1khz/K001/N09_M07_F10_K001_1_windows.npy
bearing-kd infer --input data/raw/paderborn/K001/N09_M07_F10_K001_1.mat
```

`infer`는 각 1초 윈도우의 logits와 라벨을 반환합니다. NPY는 전처리된 `[N,4000]` 또는 `[4000]`이어야 합니다. MAT 입력은 공통 전처리를 수행합니다. `.onnx.data`는 외부 가중치이므로 `.onnx`와 같은 위치에 함께 복사해야 합니다.

## 평가 범위와 이전 데이터

**추가 발견:** 기존 Teacher의 LoRA B 행렬 24개는 모두 0이었습니다. 원래 reentrant gradient checkpointing에서 고정 embedder 뒤의 LoRA로 기울기가 전달되지 않았습니다. 새 코드는 non-reentrant 방식과 gradient 검사로 수정했습니다. 기존 체크포인트의 성능은 보존된 기존 모델 결과이며, 수정된 LoRA의 최종 성능은 전체 재학습 후 평가해야 합니다.

기존 Student는 252개 test 윈도우에서 **251개 정답(99.60%)**, 혼동행렬 `[[83,1],[0,168]]`을 재현했습니다. 동일한 K001·KA03·KA04 베어링에서 측정한 서로 다른 녹음 파일을 나눈 결과입니다. 파일 중복은 없지만 **새 베어링 개체나 현장 센서에 대한 일반화 평가가 아닙니다.** 그런 주장을 하려면 베어링 개체 단위 holdout 평가가 별도로 필요합니다.

기존 모델로 KB24를 평가하면 Teacher와 Student 모두 **0/560(0%)**이고, 전부 정상으로 판정합니다. 8차 필터로 다시 전처리한 KB24도 결과가 같아서 필터 차수 문제는 아닙니다. 참고로 KI03은 70/70을 맞혔습니다.

기존 KB 데이터는 6차 필터였고 새 공통 전처리는 전 종류에 8차 필터를 씁니다. 기존 데이터·모델은 보존했습니다. 기존 모델 성능과 새로운 전처리·재학습 결과를 섞어 보고하지 마세요. 기존 Teacher NPZ는 sample ID/해시가 없어 새 학습기가 거부합니다. `cache-teacher`로 새 경로에 다시 생성하세요.

## v3 실험: 베어링 단위 holdout 평가

기존 평가(파일 단위 분할, 99.6%)가 새 베어링 성능을 보여주지 못해서 진행한 실험입니다. 결과는 `outputs/runs/v3/`에 있습니다.

| 구성 | 내용 |
|---|---|
| 전처리 | `data/processed/windows_v3` (9개 베어링, 8차 공통 전처리, 720파일/5,042윈도우. K002·KB27 파일 1개씩은 녹음이 길어 윈도우가 8개) |
| Train (408파일) / Validation (72파일) | K001·K002·K003 + KA03·KA04·KI03, 파일 단위 85/15 |
| **Test (240파일)** | **K004(정상) + KB24·KB27(고장)**, 학습에서 완전히 제외 |
| Teacher | MOMENT Base + LoRA, MPS. 20 epoch 예정이었으나 **epoch 10에서 중단** (validation이 100%에 도달) |
| Student | MOMENT Small 헤드 KD 20 epoch, 최고 validation 98.0% (epoch 20) |

| Test 베어링 | Teacher (epoch 10) | Student (PyTorch = ONNX) |
|---|---|---|
| K004 정상 | 420/560 (75.0%) | **525/560 (93.8%)** |
| KB24 고장 | **0/560 (0%)** | **0/560 (0%)** |
| KB27 고장 | 561/561 (100%) | 561/561 (100%) |
| 전체 | 981/1681 (**58.4%**) | 1086/1681 (**64.6%**) |
| 혼동행렬 | `[[420,140],[560,561]]` | `[[525,35],[560,561]]` |

- Validation(같은 베어링, 다른 파일)은 Teacher 99.4~100%, Student 96.6~98.0%였지만 test는 58~65%였습니다. **베어링 개체 오버피팅**이 확인된 것입니다.
- KI03을 학습에 추가해도 KB24는 0%로 그대로입니다. KB27은 완벽하게 맞힙니다.
- K004 오판은 운전 조건별로 편차가 큽니다. Teacher 기준 N15_M01_F10 71/140, N15_M07_F10 83/140, N09_M07_F10 128/140, N15_M07_F04 138/140입니다.
- Student가 정상 오판을 덜 합니다(140개 → 35개). 헤드만 학습해서 덜 외운 것으로 보이지만, test 정상 베어링이 1개뿐이라 확정할 수 없습니다.
- 수정된 코드로 학습한 Teacher의 LoRA B 행렬은 24/24개가 실제로 갱신됐습니다.

재현:

```bash
bearing-kd preprocess --bearings K001 K002 K003 K004 KA03 KA04 KI03 KB24 KB27 --output-dir data/processed/windows_v3
bearing-kd split --processed-dir data/processed/windows_v3 --test-bearings K004 KB24 KB27 --output data/splits/bearing_holdout_v3.json
# 실행 스크립트: outputs/runs/v3/run.sh (Teacher), outputs/runs/v3/run_student.sh (캐시 → KD → 평가 → export)
```

## Raspberry Pi 배포 검증

| 항목 | 결과 |
|---|---|
| 로드 | 성공, CPUExecutionProvider |
| 입출력 | `x_enc [1,1,512]`, `input_mask [1,512]` → `logits [1,2]`, 모두 float32 |
| 수치 검증 (K004·KB27, 운전 조건별 1파일, 56윈도우) | **PASS**: Mac 대비 최대 차이 7.2e-7, 예측 56/56 일치, 정답 55/56 |
| 속도 | **청크당 38.9 ms, 윈도우당 311 ms** (Mac CPU 2스레드는 3.2 ms/청크) |

윈도우 이동 간격이 0.5초이므로 윈도우당 500 ms 안이면 실시간 처리가 가능합니다. 지금은 CPU 약 60%를 씁니다.

```bash
# Mac → Pi 복사 (.onnx와 .onnx.data는 반드시 같은 폴더, 이름 변경 금지)
scp -r outputs/runs/v3/pi_verify <user>@<pi주소>:~/vib-onnx/
# Pi에서 실행
cd ~/vib-onnx/pi_verify && ~/vib-onnx/.venv/bin/python pi_verify.py
```

- `pi_verify_data.npz`는 전처리된 윈도우, 정답 라벨, Mac 기대 logits, 출처 이름을 묶은 NumPy 배열 묶음입니다. 위치는 자유롭게 두고 `--data`로 지정하면 됩니다.
- `.onnx` 안에는 `.onnx.data` **파일 이름**이 기록되어 있습니다. 새 모델은 `student.onnx.data`, 기존 모델은 `student_moment_chunk.onnx.data`입니다.
- **아직 검증하지 않은 것:** Pi에서 센서 원본에 직접 적용하는 전처리(필터·리샘플링)와 실시간 스트리밍 필터.

## 알려진 문제와 다음 단계

| 구분 | 내용 |
|---|---|
| 성능 | KB24 0% 검출. 새 베어링 일반화 부족 |
| 평가 | validation이 같은 베어링의 다른 파일이라 모델 선택에 쓸모가 없음. `--val-bearings`(베어링 단위 validation)가 필요 |
| 학습 | epoch별 체크포인트 미저장, `training_config.json`에 `model_dir` 미기록 |
| 결정 필요 | **현장 센서의 샘플링 레이트** (전처리 방향 결정), 추가로 쓸 베어링, KA06 부분 해제 폴더 처리 |
| 비교 실험 | LoRA 유무, KD vs 정답 라벨만으로 학습한 Small 헤드. 증류의 실제 효과가 아직 검증되지 않음 |
| 정리 | `data/processed/windows_1khz/`의 짝 없는 `N09_M07_F10_K001_1_windows.npy`(`--processed-dir windows_1khz` 평가를 깨뜨림), `.gitignore`의 `/data/interim/`와 실제 `data/initerim` 불일치, 빈 `.venv*` 폴더, 루트 `:memory:.ses` |
| 설치 | Pi는 Python 3.13인데 `requires-python`이 `<3.12`임 |

## 테스트

```bash
# 데이터·모델 다운로드 불필요
python -m unittest discover -s tests -v
# 로컬 원본/모델 필요: 전체 test 평가 + 새 export + 소규모 실제 학습
HF_HUB_OFFLINE=1 python scripts/verify_pipeline.py
```

회귀 테스트는 18개입니다. 베어링 holdout 분할 테스트가 추가됐습니다. 통합 검증은 `outputs/verification/run-*`에 결과를 저장합니다. 학습 smoke test는 클래스당 1개 윈도우, 1 epoch으로 연결 동작만 확인하며 성능 평가가 아닙니다. Raspberry Pi 결과는 위 [배포 검증](#raspberry-pi-배포-검증)을 참고하세요.

## 출처와 공개 준비

데이터 출처는 Christian Lessmeier 외, Paderborn University, Chair of Design and Drive Technology입니다. [공식 데이터 안내](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter)는 CC BY-NC 4.0 및 출처 표기를 안내합니다. MOMENT 출처와 라이선스는 각 모델 카드를 확인하세요.

프로젝트 자체의 공개 라이선스는 임의로 지정하지 않았습니다. 저장소 소유자가 코드 라이선스를 선택해야 하며 외부 데이터·모델의 조건은 별개입니다. [구조와 이전 코드 안내](docs/architecture.md), [검증 기록](docs/verification.md)을 참고하세요.
