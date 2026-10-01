"""1장 완성 코드: 개발 환경(파이썬, 패키지 버전, 연산 장치)을 점검한다.

실행 (프로젝트 루트에서, 가상환경 활성화 후):
    python docs/manual/code/ch01_check_env.py
"""

import importlib.metadata
import platform
import sys

# 이 프로젝트에서 검증한 버전 (constraints-tested.txt)
TESTED = {
    "torch": "2.14.0",
    "momentfm": "0.1.4",
    "peft": "0.21.0",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "transformers": "4.57.6",
    "huggingface-hub": "0.36.2",
    "onnx": "1.23.0",
    "onnxruntime": "1.30.0",
    "onnxscript": "0.7.2",
    "matplotlib": "3.11.2",
}


def check_python():
    print(f"Python {platform.python_version()} ({platform.system()} {platform.machine()})")
    if sys.version_info[:2] != (3, 11):
        print("  ⚠ 학습 환경은 Python 3.11 기준입니다. (추론만 하는 라즈베리파이는 3.13도 가능)")
    in_venv = sys.prefix != sys.base_prefix
    print(f"  가상환경 사용 중: {'예' if in_venv else '아니오 ← 1장의 가상환경 활성화를 먼저 하세요'}")


def check_packages():
    print("\n패키지 버전 (설치됨 / 검증됨)")
    for name, tested in TESTED.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = "없음"
        mark = "OK" if installed == tested else ("없음" if installed == "없음" else "다름")
        print(f"  [{mark:2s}] {name:16s} {installed:10s} / {tested}")


def check_device():
    import torch

    print("\n연산 장치")
    print(f"  CPU  : 항상 사용 가능 (스레드 {torch.get_num_threads()}개)")
    print(f"  CUDA : {'사용 가능 - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음 (NVIDIA GPU 전용)'}")
    print(f"  MPS  : {'사용 가능 (Apple Silicon GPU)' if torch.backends.mps.is_available() else '없음 (Apple Silicon Mac 전용)'}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.ones(2, 3, device=device)
    print(f"  → 추천 장치: {device}  (시험 계산: {x.sum().item():.0f} = 6 이면 정상)")


if __name__ == "__main__":
    check_python()
    check_packages()
    check_device()
