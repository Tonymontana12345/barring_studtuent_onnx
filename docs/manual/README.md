# 베어링 진동 AI 모델 만들기 매뉴얼

ML을 몰라도 처음부터 끝까지 따라 하면 **베어링 진동으로 정상/고장을 판정하는 모델**을 직접 만들고, 라즈베리파이 5에서 실행할 수 있도록 만든 Jupyter 매뉴얼입니다.

- 챕터마다 노트북 1개: **개념 설명 → 설명마다 코드 셀 → 실행 결과**
- 챕터 끝에는 그 챕터 내용을 한 파일로 모은 **완성 코드**(`code/chXX_*.py`). 노트북 없이 터미널에서도 실행됩니다.
- 완성 코드는 `src/bearing_kd`의 실제 프로젝트 코드를 읽기 쉽게 줄인 것이고, 같은 결과를 냅니다(4장, 6장에서 직접 비교).

## 시작하기

```bash
cd vib-onnx
source .venv-uv/bin/activate      # 환경 만들기는 1장
jupyter lab docs/manual
```

노트북을 열면 **맨 위 [준비 셀]부터 위에서 아래로** `Shift + Enter`로 실행하세요. 이 매뉴얼을 따라 하며 생기는 결과물은 `outputs/manual/`에 저장됩니다.

## 목차

| 상태 | 챕터 | 노트북 | 완성 코드 | 실행 시간 |
|---|---|---|---|---|
| | **Part 0. 시작하기 전에** | | | |
| ✅ | 0. 이 매뉴얼로 만들 것 | [00_overview](00_overview.ipynb) | [ch00_check_project.py](code/ch00_check_project.py) | 1분 |
| ✅ | 1. 개발 환경 만들기 | [01_environment](01_environment.ipynb) | [ch01_check_env.py](code/ch01_check_env.py) | 10–30분 |
| ✅ | 2. ML 최소 개념 | [02_ml_basics](02_ml_basics.ipynb) | [ch02_ml_basics.py](code/ch02_ml_basics.py) | 5분 |
| | **Part 1. 데이터** | | | |
| ✅ | 3. Paderborn 데이터 이해하기 | [03_data](03_data.ipynb) | [ch03_inspect_data.py](code/ch03_inspect_data.py) | 5분 |
| ✅ | 4. 전처리 | [04_preprocessing](04_preprocessing.ipynb) | [ch04_preprocess.py](code/ch04_preprocess.py) | 5분 |
| ✅ | 5. 데이터 나누기 | [05_split](05_split.ipynb) | [ch05_split.py](code/ch05_split.py) | 5분 |
| ✅ | 6. Dataset과 DataLoader | [06_dataset](06_dataset.ipynb) | [ch06_dataset.py](code/ch06_dataset.py) | 5분 |
| | **Part 2. 모델** | | | |
| ✅ | 7. MOMENT 기초 모델과 모델 크기 | [07_moment](07_moment.ipynb) | [ch07_moment.py](code/ch07_moment.py) | 2분 |
| ✅ | 8. LoRA | [08_lora](08_lora.ipynb) | [ch08_lora.py](code/ch08_lora.py) | 2분 |
| ✅ | 9. Teacher 학습 | [09_train_teacher](09_train_teacher.ipynb) | [ch09_train_teacher.py](code/ch09_train_teacher.py) | 맛보기 1분 / 전체 약 3.5시간 |
| ✅ | 10. 체크포인트 | [10_checkpoint](10_checkpoint.ipynb) | [ch10_checkpoint.py](code/ch10_checkpoint.py) | 1분 |
| ✅ | 11. 지식 증류(KD) | [11_distillation](11_distillation.ipynb) | [ch11_distill.py](code/ch11_distill.py) | 맛보기 1분 / 전체 약 25분 |
| ✅ | 12. 평가 | [12_evaluate](12_evaluate.ipynb) | [ch12_evaluate.py](code/ch12_evaluate.py) | 1분 |
| | **Part 3. 배포** | | | |
| ✅ | 13. ONNX 만들기 | [13_export_onnx](13_export_onnx.ipynb) | [ch13_export_onnx.py](code/ch13_export_onnx.py) | 1분 |
| ✅ | 14. ONNX 파일 구성 들여다보기 | [14_onnx_structure](14_onnx_structure.ipynb) | [ch14_inspect_onnx.py](code/ch14_inspect_onnx.py) | 1분 |
| ✅ | 15. 라즈베리파이 배포 | [15_raspberry_pi](15_raspberry_pi.ipynb) | [ch15_pi_deploy.py](code/ch15_pi_deploy.py) | Mac 1분 + 라즈베리파이 15분 |
| ✅ | 16. 허깅페이스 공개 | [16_huggingface](16_huggingface.ipynb) | [ch16_huggingface.py](code/ch16_huggingface.py) | 10분 |
| | **부록** | | | |
| ✅ | A. 문제 해결 / B. 용어집 / C. 명령어 한 페이지 / D. 실험 기록 템플릿 | [appendix.md](appendix.md) | | |

## 챕터끼리의 연결

각 챕터는 앞 챕터의 결과물을 씁니다. 순서대로 실행하세요.

```text
3장 원본 데이터 ─▶ 4장 outputs/manual/processed/ ─▶ 5장 outputs/manual/split_holdout.json
                ─▶ 6장 Dataset/DataLoader ─▶ 7–9장 Teacher ─▶ 10장 체크포인트
                ─▶ 11장 Student ─▶ 12장 평가 ─▶ 13–14장 ONNX ─▶ 15장 라즈베리파이 ─▶ 16장 공개
```

## 이 매뉴얼로 알 수 있는 것과 없는 것

- ✅ 데이터 준비부터 라즈베리파이 배포, 공개까지 **전체 과정을 직접 한 번** 해 볼 수 있습니다.
- ✅ 이 프로젝트가 실제로 겪은 버그와 함정(LoRA 기울기, 체크포인트 덮어쓰기, 평가 설계, ONNX 경로 노출)을 재현하며 배웁니다.
- ⚠ 맛보기 모델과 현재 프로젝트 모델은 **처음 보는 일부 고장 베어링을 놓칩니다**(12장). 이 매뉴얼은 "좋은 모델을 만드는 법"이 아니라 "모델을 만들고 올바르게 평가하는 과정"을 다룹니다.

## 환경

macOS (Apple Silicon), Python 3.11, torch 2.14.0, momentfm 0.1.4, peft 0.21.0, onnx 1.23.0, onnxruntime 1.30.0 기준으로 **모든 노트북 셀을 실행해 검증**했습니다. 라즈베리파이 단계는 Raspberry Pi 5(Python 3.13)에서 검증한 결과를 실었습니다. 그림 안의 글자는 운영체제별 한글 폰트 문제를 피하려고 영어로 표기했습니다.
