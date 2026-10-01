---
license: cc-by-nc-4.0
library_name: onnx
base_model: AutonLab/MOMENT-1-small
tags:
  - onnx
  - time-series-classification
  - bearing-fault-diagnosis
  - vibration
  - knowledge-distillation
  - paderborn
  - raspberry-pi
  - anomaly-detection
---

# MOMENT-Small Bearing Vibration Anomaly Classifier (ONNX, experimental)

[English](#english) | [한국어](#한국어)

<a id="english"></a>

## English

Binary normal/fault classifier for single-axis bearing vibration, built on [AutonLab/MOMENT-1-small](https://huggingface.co/AutonLab/MOMENT-1-small) and exported to ONNX for Raspberry Pi 5 CPU inference. The Student's classification head was trained by knowledge distillation from a Teacher ([AutonLab/MOMENT-1-base](https://huggingface.co/AutonLab/MOMENT-1-base) with LoRA + head).

> **Experimental model.** A vibration anomaly analysis model that runs on Raspberry Pi. Additional training may be required depending on the application environment.

### Files

```text
onnx/student.onnx              # graph (fixed batch 1, one 512-sample chunk)
onnx/student.onnx.data         # external weights — keep next to student.onnx, do not rename
pytorch/student_kd_best.pt     # Student checkpoint (MOMENT-1-small, head trained; epoch 20)
pytorch/teacher_lora_best.pt   # Teacher checkpoint (MOMENT-1-base + LoRA q/v r=8 + head; epoch 10)
pytorch/*_training_config.json # hyperparameters, package versions, data fingerprints
pytorch/*_history.json         # per-epoch training loss and validation metrics
verify/pi_verify.py            # parity check (numpy + onnxruntime only)
verify/pi_verify_data.npz      # 56 preprocessed windows + reference logits
verify/bearing_holdout_v3.json # train/validation/test recording lists used for this model
```

The PyTorch checkpoints contain the full model `state_dict` plus optimizer state, not adapter-only weights. Loading them requires the MOMENT model classes (`momentfm==0.1.4`) and the same LoRA configuration.

### Model

| | Teacher | Student (this ONNX) |
|---|---|---|
| Backbone | MOMENT-1-base, frozen | MOMENT-1-small, frozen |
| Trained parts | LoRA (q, v; r=8, α=16, dropout 0.05) + linear head | linear head only (512 → 2) |
| Trainable parameters | 296,450 | 1,026 |
| Objective | chunk-level cross-entropy | 0.5 × cross-entropy + 0.5 × KL(teacher ‖ student), T = 2 |

### Input contract

| Item | Value |
|---|---|
| Source signal | `vibration_1`, single axis, 64 kHz |
| Filter | 1 kHz low-pass, 8th-order Butterworth, zero-phase (`sosfiltfilt`) |
| Resampling | 4 kHz |
| Window / stride | 4,000 samples (1 s) / 2,000 samples |
| Chunking | 8 chunks × 512 samples; the last 96 samples are zero-padded with `input_mask = 0` |
| Normalization | none outside the model (MOMENT's RevIN is inside the graph) |
| ONNX inputs | `x_enc` float32 `[1, 1, 512]`, `input_mask` float32 `[1, 512]` |
| ONNX output | `logits` float32 `[1, 2]` |
| Window decision | **mean of the 8 chunk logits**, then argmax (0 = Normal, 1 = Fault) |

The 4 kHz rate was chosen for Raspberry Pi speed. The zero-phase filter processes whole recordings; a causal streaming filter is not included.

### Usage

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("onnx/student.onnx", providers=["CPUExecutionProvider"])

def predict_window(window):
    """window: preprocessed float32 array of shape (4000,) — see Input contract."""
    window = np.asarray(window, dtype=np.float32)
    assert window.shape == (4000,) and np.isfinite(window).all()
    chunks = np.pad(window, (0, 96)).reshape(8, 1, 512)
    masks = (np.arange(4096) < 4000).astype(np.float32).reshape(8, 512)
    logits = np.mean([
        session.run(["logits"], {"x_enc": chunks[i:i + 1], "input_mask": masks[i:i + 1]})[0][0]
        for i in range(8)
    ], axis=0)
    return logits, int(np.argmax(logits))  # 0 = Normal, 1 = Fault
```

The input must already be preprocessed; do not feed raw sensor values.

Check that your runtime reproduces the reference outputs:

```bash
python verify/pi_verify.py --threads 4
```

### Training data

Paderborn bearing data. Training and validation use different recordings of K001, K002, K003 (normal) and KA03, KA04, KI03 (fault), split 85/15 by recording. K004, KB24 and KB27 were held out from training.

**Note:** Additional training may be required depending on the application environment.

### Raspberry Pi 5 performance

| Item | Result |
|---|---|
| Device | Raspberry Pi 5 |
| Runtime | onnxruntime, CPUExecutionProvider, Python 3.13 |
| Parity vs macOS reference (`verify/`) | PASS — max logit difference 7.2e-7, 56/56 predictions identical |
| Latency | 38.9 ms per chunk, ~311 ms per 1-s window |

Windows advance every 0.5 s, so ~311 ms per window leaves headroom for real-time use. Latency may vary with thread settings and thermal conditions.

### Training details

- Hyperparameters: AdamW, learning rate 1e-4, batch 16, seed 42. Teacher: 10 epochs. Student: 20 epochs, best validation at epoch 20.
- Hardware: Apple Silicon (PyTorch MPS backend). Versions: torch 2.14.0, momentfm 0.1.4, peft 0.21.0, numpy 2.4.6.
- The Teacher uses non-reentrant gradient checkpointing so that gradients reach the LoRA weights while the embedder is frozen.

### Attribution and license

- **Data:** Paderborn University Bearing DataCenter — C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro, *"Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification"*, PHM Society European Conference, 2016. [Official page](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter). The data is provided under **CC BY-NC 4.0**, so these weights are released for **non-commercial use** with attribution.
- **Base models:** [AutonLab/MOMENT-1-small](https://huggingface.co/AutonLab/MOMENT-1-small), [AutonLab/MOMENT-1-base](https://huggingface.co/AutonLab/MOMENT-1-base). See their model cards for license terms.

---

<a id="한국어"></a>

## 한국어

[AutonLab/MOMENT-1-small](https://huggingface.co/AutonLab/MOMENT-1-small) 기반의 단일 축 베어링 진동 정상/고장 분류 모델입니다. Raspberry Pi 5 CPU 추론용 ONNX로 변환했습니다. Student의 분류 헤드는 Teacher([AutonLab/MOMENT-1-base](https://huggingface.co/AutonLab/MOMENT-1-base) + LoRA + 헤드)의 예측을 이용한 지식 증류(knowledge distillation)로 학습했습니다.

> **실험용 모델입니다.** 라즈베리파이에 적용 가능한 진동 이상치 분석 모델이며, 상황에 따라 추가 학습이 필요합니다.

### 파일

```text
onnx/student.onnx              # 그래프 (배치 1 고정, 512 샘플 청크 1개)
onnx/student.onnx.data         # 외부 가중치 — student.onnx와 같은 폴더에 두고 이름 변경 금지
pytorch/student_kd_best.pt     # Student 체크포인트 (MOMENT-1-small, 헤드 학습, epoch 20)
pytorch/teacher_lora_best.pt   # Teacher 체크포인트 (MOMENT-1-base + LoRA q/v r=8 + 헤드, epoch 10)
pytorch/*_training_config.json # 하이퍼파라미터, 패키지 버전, 데이터 지문
pytorch/*_history.json         # epoch별 학습 손실과 validation 지표
verify/pi_verify.py            # 수치 검증 스크립트 (numpy + onnxruntime만 필요)
verify/pi_verify_data.npz      # 전처리된 윈도우 56개 + 기준 logits
verify/bearing_holdout_v3.json # 이 모델에 사용한 train/validation/test 파일 목록
```

PyTorch 체크포인트는 어댑터만이 아니라 전체 모델 `state_dict`와 optimizer 상태를 포함합니다. 로딩하려면 MOMENT 모델 클래스(`momentfm==0.1.4`)와 같은 LoRA 구성이 필요합니다.

### 모델

| | Teacher | Student (이 ONNX) |
|---|---|---|
| 백본 | MOMENT-1-base, 고정 | MOMENT-1-small, 고정 |
| 학습 대상 | LoRA (q, v; r=8, α=16, dropout 0.05) + 선형 헤드 | 선형 헤드만 (512 → 2) |
| 학습 파라미터 수 | 296,450 | 1,026 |
| 손실 | 청크 단위 cross-entropy | 0.5 × cross-entropy + 0.5 × KL(teacher ‖ student), T = 2 |

### 입력 형식

| 항목 | 값 |
|---|---|
| 원본 신호 | `vibration_1`, 단일 축, 64 kHz |
| 필터 | 1 kHz 저역통과, Butterworth 8차, 양방향(`sosfiltfilt`) |
| 리샘플링 | 4 kHz |
| 윈도우 / 이동 간격 | 4,000 샘플(1초) / 2,000 샘플 |
| 청크 | 512 샘플 × 8개, 마지막 96 샘플은 0 패딩 + `input_mask = 0` |
| 정규화 | 모델 밖에서는 하지 않음 (MOMENT RevIN이 그래프 안에 포함) |
| ONNX 입력 | `x_enc` float32 `[1, 1, 512]`, `input_mask` float32 `[1, 512]` |
| ONNX 출력 | `logits` float32 `[1, 2]` |
| 윈도우 판정 | **8개 청크 logits 평균** 후 argmax (0 = 정상, 1 = 고장) |

4 kHz는 라즈베리파이 추론 속도를 고려해 선택했습니다. 양방향 필터는 녹음 구간 단위로 처리하며, 실시간 스트리밍용 인과 필터는 포함하지 않습니다.

### 사용법

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("onnx/student.onnx", providers=["CPUExecutionProvider"])

def predict_window(window):
    """window: 전처리를 마친 (4000,) float32 배열 — 입력 형식 참고."""
    window = np.asarray(window, dtype=np.float32)
    assert window.shape == (4000,) and np.isfinite(window).all()
    chunks = np.pad(window, (0, 96)).reshape(8, 1, 512)
    masks = (np.arange(4096) < 4000).astype(np.float32).reshape(8, 512)
    logits = np.mean([
        session.run(["logits"], {"x_enc": chunks[i:i + 1], "input_mask": masks[i:i + 1]})[0][0]
        for i in range(8)
    ], axis=0)
    return logits, int(np.argmax(logits))  # 0 = 정상, 1 = 고장
```

입력은 반드시 전처리를 마친 값이어야 합니다. 센서 원본 값을 그대로 넣지 마세요.

실행 환경이 기준 출력을 재현하는지 확인:

```bash
python verify/pi_verify.py --threads 4
```

### 학습 데이터

Paderborn 베어링 데이터를 사용했습니다. 학습과 검증에는 K001, K002, K003(정상)과 KA03, KA04, KI03(고장)의 서로 다른 녹음 파일을 85/15로 나눠 사용했습니다. K004, KB24, KB27은 학습에서 제외했습니다.

**참고:** 상황에 따라 추가 학습이 필요합니다.

### Raspberry Pi 5 성능

| 항목 | 결과 |
|---|---|
| 장치 | Raspberry Pi 5 |
| 런타임 | onnxruntime, CPUExecutionProvider, Python 3.13 |
| macOS 기준값과 수치 비교 (`verify/`) | PASS — logits 최대 차이 7.2e-7, 예측 56/56 일치 |
| 추론 시간 | 청크당 38.9 ms, 1초 윈도우당 약 311 ms |

윈도우가 0.5초마다 새로 만들어지므로 윈도우당 약 311 ms면 실시간 처리에 여유가 있습니다. 스레드 설정과 발열 상태에 따라 시간은 달라질 수 있습니다.

### 학습 정보

- 하이퍼파라미터: AdamW, learning rate 1e-4, batch 16, seed 42. Teacher 10 epoch, Student 20 epoch (최고 validation은 epoch 20).
- 하드웨어: Apple Silicon (PyTorch MPS). 버전: torch 2.14.0, momentfm 0.1.4, peft 0.21.0, numpy 2.4.6.
- Teacher는 embedder를 고정한 상태에서도 LoRA에 기울기가 전달되도록 non-reentrant gradient checkpointing을 사용합니다.

### 출처와 라이선스

- **데이터:** Paderborn University Bearing DataCenter — C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro, *"Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification"*, PHM Society European Conference, 2016. [공식 페이지](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter). 데이터가 **CC BY-NC 4.0**으로 제공되므로 이 가중치도 출처 표기 조건의 **비상업적 용도**로 공개합니다.
- **기반 모델:** [AutonLab/MOMENT-1-small](https://huggingface.co/AutonLab/MOMENT-1-small), [AutonLab/MOMENT-1-base](https://huggingface.co/AutonLab/MOMENT-1-base). 라이선스 조건은 각 모델 카드를 확인하세요.
