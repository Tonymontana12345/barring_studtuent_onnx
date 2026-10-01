"""3장 완성 코드: Paderborn .mat 파일을 열어 구조를 보고, 진동 신호를 꺼내 그린다.

실행 (프로젝트 루트에서):
    python docs/manual/code/ch03_inspect_data.py
    python docs/manual/code/ch03_inspect_data.py --normal K001 --fault KA04 --condition N15_M07_F10
결과 그림: outputs/manual/ch03_waveforms.png
"""

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data/raw/paderborn"
RAW_FS = 64_000  # vibration_1의 샘플링 레이트 (1초에 64,000번 기록)

CONDITIONS = {  # 파일 이름 앞부분 = 운전 조건
    "N15_M07_F10": "1500 rpm, 부하 토크 0.7 Nm, 반경 방향 힘 1000 N (기준 조건)",
    "N09_M07_F10": "900 rpm, 0.7 Nm, 1000 N (저속)",
    "N15_M01_F10": "1500 rpm, 0.1 Nm, 1000 N (저부하)",
    "N15_M07_F04": "1500 rpm, 0.7 Nm, 400 N (약한 반경 방향 힘)",
}


def bearing_label(code):
    """베어링 코드 → 라벨. K001처럼 K+숫자3자리는 정상(0), KA/KB/KI+숫자2자리는 고장(1)."""
    if re.fullmatch(r"K\d{3}", code):
        return 0
    if re.fullmatch(r"K[ABI]\d{2}", code):
        return 1
    raise ValueError(f"알 수 없는 베어링 코드: {code}")


def load_record(path):
    """.mat 파일 안에는 파일 이름과 같은 이름의 구조체 하나가 들어 있다."""
    path = Path(path)
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    return mat[path.stem]


def list_channels(record):
    """Y 안의 측정 채널 이름, 길이, 기록 방식(Raster)을 돌려준다."""
    channels = []
    for item in np.atleast_1d(record.Y):
        channels.append((str(item.Name), np.asarray(item.Data).size, str(item.Raster)))
    return channels


def extract_vibration(path):
    record = load_record(path)
    for item in np.atleast_1d(record.Y):
        if str(item.Name) == "vibration_1":
            return np.asarray(item.Data, dtype=np.float32).reshape(-1)
    raise ValueError(f"vibration_1 없음: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", default="K001")
    parser.add_argument("--fault", default="KA04")
    parser.add_argument("--condition", default="N15_M07_F10", choices=list(CONDITIONS))
    args = parser.parse_args()

    files = {code: RAW_DIR / code / f"{args.condition}_{code}_1.mat" for code in (args.normal, args.fault)}
    first = files[args.normal]
    print(f"파일: {first.relative_to(ROOT)}")
    print(f"운전 조건 {args.condition}: {CONDITIONS[args.condition]}\n")

    record = load_record(first)
    print("구조체 필드:", record._fieldnames)
    print("\n측정 채널 (Y)")
    for name, length, raster in list_channels(record):
        print(f"  {name:22s} 샘플 {length:>7,}개  ({raster})")

    signals = {code: extract_vibration(path) for code, path in files.items()}
    print()
    for code, x in signals.items():
        print(f"{code} (라벨 {bearing_label(code)}): {x.size:,} 샘플 = {x.size / RAW_FS:.2f}초, "
              f"평균 {x.mean():+.4f}, 표준편차 {x.std():.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = int(0.02 * RAW_FS)  # 앞부분 0.02초만
    t = np.arange(n) / RAW_FS * 1000
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
    for ax, (code, x) in zip(axes, signals.items()):
        ax.plot(t, x[:n], linewidth=0.7)
        # 그림 글자는 영어로 (운영체제마다 한글 폰트가 달라 깨질 수 있음)
        ax.set_title(f"{code} ({'normal' if bearing_label(code) == 0 else 'fault'}) - vibration_1")
        ax.set_ylabel("vibration")
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    out = ROOT / "outputs/manual/ch03_waveforms.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n그림 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
