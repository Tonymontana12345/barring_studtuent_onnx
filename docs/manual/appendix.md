# 부록

- [A. 문제 해결](#a-문제-해결): 이 프로젝트와 매뉴얼을 만들며 실제로 겪은 문제
- [B. 용어집](#b-용어집)
- [C. 명령어 한 페이지](#c-명령어-한-페이지)
- [D. 실험 기록 템플릿](#d-실험-기록-템플릿)

---

## A. 문제 해결

모두 이 프로젝트에서 **실제로 겪은** 문제입니다. 증상(에러 메시지)으로 찾아보세요.

### 환경·설치

| 증상 | 원인 | 해결 | 관련 장 |
|---|---|---|---|
| `ModuleNotFoundError: No module named 'onnxruntime'` (또는 torch, numpy) | 가상환경을 켜지 않고 시스템 파이썬으로 실행 | `source .venv/bin/activate` 후 실행하거나 `.venv/bin/python 스크립트.py`로 직접 실행 | 1, 15 |
| `pip install` 중 numpy·transformers·huggingface-hub 버전 충돌 | `momentfm 0.1.4`가 오래된 버전을 정확히 고정 | `uv pip install -r requirements-lock.txt --override constraints-tested.txt` | 1 |
| `pip check`가 충돌 3개를 경고 | override로 설치했기 때문 (정상) | 무시. 검증된 조합임 | 1 |
| 라즈베리파이에서 `pip install -e .` 거부 | `pyproject.toml`이 Python `<3.12`만 허용, 라즈베리파이는 3.13 | 추론만 필요하면 `pip install numpy onnxruntime scipy` + 독립 스크립트(`pi_infer.py`) | 15 |
| matplotlib 그림에서 한글이 네모로 깨짐 (`Glyph ... missing`) | 기본 폰트에 한글 없음 | 그림 글자는 영어로 쓰거나 한글 폰트를 따로 지정 | 3 |

### 학습

| 증상 | 원인 | 해결 | 관련 장 |
|---|---|---|---|
| **LoRA B 행렬이 학습 후에도 전부 0**, 에러는 없음 | gradient checkpointing이 reentrant 방식이라 LoRA에 기울기가 안 감 | `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` + 매 step `validate_gradients` | 8 |
| `RuntimeError: 기울기 없음: ...lora_A...` | 위 버그를 검사가 잡아낸 것 | 위와 같음 | 8 |
| `Expected all tensors to be on the same device, ... mps:0 and cpu` | mask 같은 텐서를 CPU에 만들고 모델은 GPU(MPS)에 있음 | `torch.arange(..., device=x.device)`처럼 입력과 같은 장치에 만든다 | 6, 9 |
| MPS에서 `NotImplementedError: ... not currently implemented for the MPS device` | MPS가 지원하지 않는 연산 | `export PYTORCH_ENABLE_MPS_FALLBACK=1` | 1, 9 |
| 학습이 예상보다 3–4배 느림 | LoRA 버그 수정 후 역전파가 인코더 전체를 지나가고, checkpointing이 순전파를 한 번 더 함 | 정상. Mac이면 `--device mps` | 8, 9 |
| 작은 데이터로 학습했더니 loss가 0.69에서 안 내려감 (전부 한쪽으로 예측) | epoch당 step이 너무 적어(6번) learning rate 1e-4로는 거의 안 움직임 | 맛보기는 1e-3, head만 학습하면 1e-2. 전체 데이터는 1e-4 | 9, 11 |
| 자리를 비운 사이 학습이 멈춤 | Mac 잠자기, 또는 앱·노트북 종료 | 터미널에서 실행, 전원 연결, `caffeinate -i -s -w <PID>` | 9 |
| `ValueError: ... fingerprint mismatch` / 캐시를 만든 뒤 데이터가 바뀜 | Teacher 캐시와 현재 데이터(내용·순서)가 다름 | `cache-teacher`(매뉴얼 11장)를 다시 실행 | 6, 11 |

### 체크포인트·평가

| 증상 | 원인 | 해결 | 관련 장 |
|---|---|---|---|
| 원하던 epoch의 모델이 사라짐 | "정확도가 같거나 높으면 저장" 규칙이 100%에서 계속 덮어씀 | epoch별로 따로 저장 + (정확도, −loss)로 비교 | 9, 10 |
| 덜 학습된 epoch 1이 최고 모델로 남음 | "더 높을 때만 저장" 규칙인데 첫 epoch부터 100% | (정확도, −loss)로 비교 | 9 |
| `Error(s) in loading state_dict ... Missing key(s)` | 모델 구조(LoRA 유무·설정)가 체크포인트와 다름 | 체크포인트의 `lora_config`로 같은 구조를 만든 뒤 `strict=True`로 로딩 | 10 |
| **validation 100%인데 처음 보는 베어링에서 크게 틀림** | validation이 train과 같은 베어링의 다른 파일이라 너무 쉬움, 베어링 개체를 외움 | 베어링 단위 holdout, 학습 베어링 추가, validation도 베어링 단위로 | 5, 12 |

### ONNX·배포

| 증상 | 원인 | 해결 | 관련 장 |
|---|---|---|---|
| `UnsupportedOperatorError: Exporting the operator 'aten::nanmean'` | ONNX에 nanmean 없음 (옛 TorchScript exporter) | `dynamo=True` 사용 또는 RevIN을 기본 연산으로 교체 | 13 |
| Jupyter 커널이 export 셀에서 멈춤 | 옛 exporter가 실패하며 그래프 수천 줄을 stderr로 출력 | 별도 프로세스(`subprocess`)로 실행하고 필요한 줄만 출력 | 13 |
| `Unsupported model IR version: 14, max supported IR version: 13` | `onnx` 라이브러리가 `onnxruntime`보다 새 파일 형식으로 저장 | `helper.make_model(..., ir_version=10)` 또는 두 라이브러리 버전 맞추기 | 14 |
| `NO_SUCHFILE : Load model from models/xxx.onnx failed` | 경로가 틀림 (실행 위치 기준 상대 경로) 또는 파일을 안 옮김 | `ls`로 확인, 스크립트 위치 기준 경로 `Path(__file__).parent / ...` 사용 | 15 |
| `.onnx`는 있는데 로딩 실패 (외부 데이터 관련 에러) | `.onnx.data`가 없거나 이름이 바뀜. `.onnx` 안에 파일 이름이 적혀 있음 | 두 파일을 같은 폴더에, 원래 이름 그대로 | 14, 15 |
| 라즈베리파이 결과가 Mac과 `1e-7` 수준으로 다름 | CPU 종류에 따른 부동소수점 반올림 차이 | 정상. 허용 오차(1e-4) 안이고 예측이 같으면 통과 | 15 |
| 공개한 ONNX에 `/Users/이름/...`이 들어 있음 | exporter가 노드마다 stack trace 메타데이터를 남김 | 공개 전 `metadata_props`에서 `stack_trace` 삭제 + 파일 전체 문자열 검사 | 14, 16 |
| 허깅페이스에서 옛 파일을 지웠는데 기록에 남음 | 허깅페이스 저장소는 git 기반 | `super_squash_history` (되돌릴 수 없음). 처음부터 점검 후 업로드 | 16 |

---

## B. 용어집

| 용어 | 뜻 | 처음 나오는 장 |
|---|---|---|
| 가상환경 (venv) | 프로젝트 전용 파이썬 + 패키지 상자 | 1 |
| MPS / CUDA | Apple / NVIDIA GPU를 PyTorch에서 부르는 이름 | 1 |
| 텐서 (tensor) | PyTorch의 다차원 배열 | 1 |
| 분류 (classification) | 입력을 정해진 종류 중 하나로 고르는 문제 | 2 |
| train / validation / test | 학습용 / 중간 점검·모델 선택용 / 최종 평가용 데이터 | 2 |
| logits | 모델이 내는 날것의 점수 | 2 |
| softmax | logits를 합이 1인 확률로 바꾸는 함수 | 2 |
| loss (cross-entropy) | 예측이 정답과 얼마나 다른지. 정답 확률 p이면 −log p | 2 |
| batch / epoch | 한 번에 묶는 샘플 수 / 학습 데이터 한 바퀴 | 2 |
| 기울기 (gradient) | 가중치를 조금 바꿀 때 loss가 변하는 정도 | 2 |
| optimizer (SGD, AdamW) | 기울기를 보고 가중치를 고치는 규칙 | 2, 9 |
| learning rate | 한 번에 가중치를 얼마나 크게 고칠지 | 2 |
| 오버피팅 | 규칙 대신 학습 데이터를 외워서 새 데이터에 약해지는 것 | 2 |
| 혼동행렬 | 실제/예측 조합별 개수 표 | 2 |
| 오경보 / 놓침 | 정상을 고장이라 함 / 고장을 정상이라 함 | 2, 12 |
| 샘플링 레이트 | 1초에 값을 기록하는 횟수 (Hz) | 4 |
| 나이퀴스트 주파수 | 샘플링 레이트의 절반. 담을 수 있는 최대 주파수 | 4 |
| 에일리어싱 | 너무 빠른 진동이 가짜 느린 진동으로 둔갑하는 현상 | 4 |
| 저역통과 필터 | 정해진 주파수보다 빠른 성분을 지우는 필터 | 4 |
| 윈도우 / stride | 신호를 자른 조각 / 조각을 옮기는 간격 | 4 |
| 전처리 계약 | 학습·추론에서 똑같이 지켜야 하는 전처리 규칙 | 4 |
| 데이터 누수 | test 정보가 학습에 새어 들어가 성능이 부풀려지는 것 | 5 |
| holdout | 특정 데이터를 학습에서 통째로 빼 두는 것 | 5 |
| seed | 무작위의 시작점. 같으면 결과가 재현됨 | 5 |
| Dataset / DataLoader | 샘플 하나를 꺼내는 상자 / 묶고 섞어서 꺼내 주는 도구 | 6 |
| 청크 / mask | MOMENT 입력 조각(512) / 진짜(1)·패딩(0) 표시 | 6 |
| 데이터 지문 (fingerprint) | 데이터 내용·순서로 만든 해시. 바뀌면 달라짐 | 6 |
| foundation model | 대량의 데이터로 미리 학습한 범용 모델 | 7 |
| RevIN | 청크마다 평균 0, 표준편차 1로 맞추는 정규화 | 7 |
| 패치 (patch) | 연속된 샘플 8개 묶음. Transformer의 한 단어 역할 | 7 |
| Transformer / attention | 패치들이 서로를 참고해 특징을 다듬는 구조 | 7, 8 |
| head | 특징을 받아 최종 점수를 내는 마지막 층 | 7 |
| freeze | 학습 중 가중치를 고치지 않게 고정 | 7 |
| LoRA (r, alpha) | 고정된 행렬 옆에 작은 보정 행렬 B·A를 붙여 그것만 학습 | 8 |
| gradient checkpointing | 중간 결과를 버렸다가 역전파 때 다시 계산해 메모리를 아끼는 기법 | 8 |
| 체크포인트 / state_dict | 학습된 모델을 저장한 파일 / "층 이름 → 텐서" 딕셔너리 | 10 |
| 지식 증류 (KD) | 큰 모델(Teacher)의 판단을 작은 모델(Student)이 따라 배우게 하는 방법 | 11 |
| temperature | 확률 분포를 부드럽게 펼치는 나눗수 | 11 |
| KL divergence | 두 확률 분포의 차이 | 11 |
| ONNX / onnxruntime | 모델 공통 파일 형식 / ONNX 실행기 | 13 |
| opset / IR 버전 | ONNX 연산 목록 버전 / 파일 형식 버전 | 13, 14 |
| 노드 / initializer | ONNX 그래프의 연산 하나 / 미리 정해진 값(가중치) | 14 |
| external data | 큰 가중치를 `.onnx.data` 별도 파일에 저장하는 방식 | 13, 14 |
| 모델 카드 | 허깅페이스 저장소의 README. 모델 설명서 | 16 |

---

## C. 명령어 한 페이지

### 매뉴얼 스크립트 (프로젝트 루트에서, `source .venv-uv/bin/activate` 후)

```bash
# 0–2장 점검·기초
python docs/manual/code/ch00_check_project.py
python docs/manual/code/ch01_check_env.py
python docs/manual/code/ch02_ml_basics.py
# 3–6장 데이터
python docs/manual/code/ch03_inspect_data.py
python docs/manual/code/ch04_preprocess.py --overwrite
python docs/manual/code/ch05_split.py --test-bearings K002 KA04
python docs/manual/code/ch06_dataset.py
# 7–12장 모델 (맛보기)
python docs/manual/code/ch07_moment.py --size base
python docs/manual/code/ch08_lora.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python docs/manual/code/ch09_train_teacher.py
python docs/manual/code/ch10_checkpoint.py
PYTORCH_ENABLE_MPS_FALLBACK=1 python docs/manual/code/ch11_distill.py
python docs/manual/code/ch12_evaluate.py
# 13–16장 배포
python docs/manual/code/ch13_export_onnx.py --overwrite
python docs/manual/code/ch14_inspect_onnx.py
python docs/manual/code/ch15_pi_deploy.py
python docs/manual/code/ch16_huggingface.py
```

### 실제 프로젝트 CLI (`bearing-kd`)

```bash
bearing-kd preprocess --bearings K001 K002 K003 K004 KA03 KA04 KI03 KB24 KB27 --output-dir data/processed/windows_v3
bearing-kd split --processed-dir data/processed/windows_v3 --test-bearings K004 KB24 KB27 --output data/splits/bearing_holdout_v3.json
bearing-kd train-teacher --split data/splits/bearing_holdout_v3.json --output-dir outputs/runs/v3/teacher --device mps
bearing-kd cache-teacher --split data/splits/bearing_holdout_v3.json --checkpoint outputs/runs/v3/teacher/teacher_lora_best.pt --output-dir outputs/runs/v3/cache --device mps
bearing-kd train-student --split data/splits/bearing_holdout_v3.json --cache-dir outputs/runs/v3/cache --output-dir outputs/runs/v3/student --device mps
bearing-kd evaluate-student --split data/splits/bearing_holdout_v3.json --checkpoint outputs/runs/v3/student/student_kd_best.pt
bearing-kd export --checkpoint outputs/runs/v3/student/student_kd_best.pt --output outputs/runs/v3/onnx/student.onnx
bearing-kd infer --input data/raw/paderborn/K001/N09_M07_F10_K001_1.mat --model outputs/runs/v3/onnx/student.onnx
```

공통 옵션 `--root`, `--threads`는 하위 명령 **앞에** 씁니다 (`bearing-kd --threads 8 train-teacher ...`).

### 라즈베리파이

```bash
python3 -m venv ~/vib-onnx/.venv && ~/vib-onnx/.venv/bin/python -m pip install numpy onnxruntime scipy
scp -r outputs/manual/pi_bundle <user>@<pi-address>:~/vib-onnx/       # Mac에서
cd ~/vib-onnx/pi_bundle && ~/vib-onnx/.venv/bin/python pi_infer.py verify
```

---

## D. 실험 기록 템플릿

실험마다 아래를 복사해 채우세요. **결과가 나쁜 실험도 기록**해야 같은 실수를 반복하지 않습니다.

```markdown
## 실험 이름: (예: v4 – validation도 베어링 단위)
- 날짜 / 작성자:
- 목적 (무엇을 확인하려고?):

### 데이터
- 전처리: 샘플링 __ kHz, 저역통과 __ Hz, 필터 __차, 윈도우 __/__ (preprocessing.json 경로: )
- train 베어링:            validation 베어링:            test 베어링:
- split 파일 / SHA-256:

### 모델·학습
- Teacher: MOMENT-1-____ + LoRA(r=__, alpha=__, 대상=__)   epoch __, lr __, batch __, seed __
- Student: MOMENT-1-____ head, KD T=__, 정답 비중 __        epoch __, lr __
- 장치 / 소요 시간:
- LoRA B 0이 아닌 개수: __/__   (0이면 학습 안 된 것!)
- 최고 모델 선택 기준 / 선택된 epoch:

### 결과 (test)
| 베어링 | 정답/전체 | 정확도 | 비고 |
|---|---|---|---|
|  |  |  |  |
- 오경보율 __ %, 놓침률 __ %
- validation과 test 차이:

### ONNX·배포
- ONNX 파일 / 검증 최대 차이:           stack trace 제거: 예/아니오
- 라즈베리파이 결과 (PASS/FAIL, 윈도우당 ms):

### 해석과 다음 할 일
-
```
