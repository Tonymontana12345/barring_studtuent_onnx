# 실행 검증 기록

검증일: 2026-09-25. 환경: macOS 26.2 ARM64, Python 3.11.16, CPU 2 threads. 패키지 버전은 `constraints-tested.txt`, 전체 해석된 의존성은 `requirements-lock.txt`에 기록했습니다.

## 실제 실행 결과

| 검증 | 결과 |
|---|---|
| 모델·데이터가 필요 없는 회귀 테스트 | 17개 통과 |
| 기존 Student PyTorch test | 251/252, 99.6032%, 혼동행렬 `[[83,1],[0,168]]` |
| 기존 ONNX 전체 test | 251/252, PyTorch와 모든 예측 일치 |
| 기존 ONNX vs PyTorch 윈도우 logits 최대 차이 | 3.5763e-7 |
| 기존 Teacher test | 252/252, 혼동행렬 `[[84,0],[0,168]]` |
| 기존 체크포인트 strict 로딩 | Teacher/Student 모두 통과 |
| 새 ONNX export | ONNX checker 및 12개 입력 비교 통과 |
| 새 export vs 원본 PyTorch 최대 차이 | 1.1921e-6 |
| 실제 K001·KA04·KB24·KI03 MAT 전처리 | 각 7개 윈도우, 라벨 0/1/1/1 |
| 새 공통 전처리 vs 기존 K001·KA04 샘플 | 최대 차이 0 |
| Teacher 1 epoch 소규모 학습 | 유한한 loss/gradient, 체크포인트 생성 |
| Teacher LoRA B 행렬 갱신 | 24/24개가 0 초기값에서 실제 갱신 |
| Teacher 캐시 → Student 1 epoch KD | 완료, 엄격한 체크포인트 재로딩 및 평가 통과 |
| 원본 MAT → ONNX CLI | 프로젝트 밖 `/tmp`에서도 실행, K001 7윈도우 모두 정상 예측 |
| 전처리·split CLI | 실제 MAT 14개 → 98윈도우, 파일 8/2/4 분할 |
| 추론 의존성 | inference/CLI import에 torch, momentfm, peft가 로드되지 않음 |
| 설치 메타데이터 | editable 설치, CLI entry point 확인 |
| 의존성 해결 | PyPI에서 uv override로 47개 패키지 해결, 기존 환경 버전과 일치 |
| 분석 도구 | MAT 검사, 원본 파형 비교, 가공 파형 비교, 스펙트럼 비교, 임베딩 추출, KB 신호 검사 모두 실행 |
| Student 학습 CLI | 실제 명령으로 1 epoch KD 및 체크포인트 저장 확인 |
| Git 제외 규칙 | 임시 Git 메타데이터로 검사: 후보 파일 31개, 약 121 KiB; 대용량 데이터/모델/환경 제외 |
| 민감정보 패턴 검사 | Git 후보 파일에서 대표 토큰·개인키 패턴 미발견 (완전한 부재 보증은 아님) |

수치 비교 허용오차는 `rtol=1e-4`, `atol=1e-5`입니다. export 입력은 무작위·영 신호·상수·마지막 96샘플 패딩 및 실제 윈도우의 8청크입니다. 파일만 만들어지는지 확인한 것이 아니라 원본 PyTorch와 ONNX 출력까지 비교했습니다.

## 검증 과정에서 추가 발견한 LoRA 문제

기존 모델 구성의 reentrant gradient checkpointing을 그대로 사용하면 LoRA 파라미터 48개 모두 `grad=None`이었습니다. 기존 Teacher 체크포인트의 LoRA B 행렬 24개도 모두 0이었습니다. 분류 헤드의 성능은 재현되지만, 해당 가중치를 LoRA가 실제 갱신된 Teacher라고 설명해서는 안 됩니다.

새 구성에서는 Teacher의 checkpointing을 non-reentrant로 설정하고 Student의 불필요한 checkpointing은 끕니다. 학습 중 모든 학습 가능 파라미터의 gradient 존재/유한성을 검사하며, 통합 검증은 저장된 LoRA B 24개가 실제로 바뀌는지도 확인합니다.

## 범위와 제외 사항

- 전체 20 epoch 재학습은 수행하지 않았습니다. 기존 모델의 test 성능 재현과 소규모 새 학습 파이프라인 검증을 구분합니다.
- 소규모 학습은 각 split에서 클래스당 1개 윈도우를 사용하는 1 epoch입니다. 이때의 정확도는 성능 지표로 사용하지 않습니다.
- 기존 모델·데이터·분할·배포 가중치는 덮어쓰지 않았습니다.
- 새 환경 전체 다운로드/설치는 수행하지 않았습니다. 기존 환경에 패키지를 editable 설치했고, 의존성은 PyPI 해결 및 dry-run으로 검증했습니다. MOMENT의 원본 메타데이터 충돌 3개는 uv override를 적용한 구성의 알려진 사항입니다.
- 모델 다운로드 스크립트의 네트워크 다운로드는 기존 로컬 모델을 보존하기 위해 실행하지 않았습니다.
- Raspberry Pi 5 하드웨어 속도, CUDA 학습, 새로운 베어링 개체 일반화 성능은 이번에 검증하지 않았습니다.
- ONNX exporter 내부 상수 계산에서 경고가 출력되었지만 ONNX checker와 실제 runtime 수치 비교는 통과했습니다.

## 재실행

```bash
source .venv-uv/bin/activate
python -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 python scripts/verify_pipeline.py
```

통합 검증은 매번 `outputs/verification/run-*`에 새 출력과 `report.json`을 생성합니다. 코드와 함께 보관할 요약은 [verification-results.json](verification-results.json)에 있습니다. 기존 검증의 소규모 체크포인트는 제품 배포용이 아닙니다.
