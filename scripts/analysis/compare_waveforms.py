from bearing_kd.config import PROJECT_ROOT

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.signal import periodogram


DATA_DIR = (PROJECT_ROOT / "data/processed/windows_1khz")

OUTPUT_DIR = (PROJECT_ROOT / "outputs/plots")

FS = 4000
CONDITION = "N09_M07_F10"

BEARING_CODES = [
    "K001",
    "KA04",
    "KB24",
]


def find_window_file(
    bearing_code,
):
    pattern = (
        f"{CONDITION}_"
        f"{bearing_code}_*_windows.npy"
    )

    paths = sorted(
        DATA_DIR.rglob(pattern)
    )

    if not paths:
        raise FileNotFoundError(
            f"{bearing_code} 파일을 찾지 못했습니다: "
            f"{pattern}"
        )

    return paths[0]


def calculate_rms(signal):
    return np.sqrt(
        np.mean(signal ** 2)
    )


def calculate_crest_factor(signal):
    rms = calculate_rms(signal)

    if rms == 0:
        return 0.0

    return np.max(
        np.abs(signal)
    ) / rms


def main():
    signals = {}

    print("[비교 파일]")

    for bearing_code in BEARING_CODES:
        path = find_window_file(
            bearing_code
        )

        windows = np.load(path)

        # 첫 번째 4000 샘플 Window 사용
        signal = windows[0]

        signals[bearing_code] = signal

        print(
            f"{bearing_code}: {path}"
        )

        print(
            f"  windows shape: "
            f"{windows.shape}"
        )

        print(
            f"  min : {signal.min():.6f}"
        )

        print(
            f"  max : {signal.max():.6f}"
        )

        print(
            f"  mean: {signal.mean():.6f}"
        )

        print(
            f"  std : {signal.std():.6f}"
        )

        print(
            f"  rms : "
            f"{calculate_rms(signal):.6f}"
        )

        print(
            f"  crest factor: "
            f"{calculate_crest_factor(signal):.6f}"
        )

    time_axis = np.arange(
        4000
    ) / FS

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(14, 10),
    )

    for row, bearing_code in enumerate(
        BEARING_CODES
    ):
        signal = signals[bearing_code]

        frequencies, power = periodogram(
            signal,
            fs=FS,
        )

        # 시간 영역 파형
        axes[row, 0].plot(
            time_axis,
            signal,
            linewidth=0.7,
        )

        axes[row, 0].set_title(
            f"{bearing_code} waveform"
        )

        axes[row, 0].set_xlabel(
            "Time (seconds)"
        )

        axes[row, 0].set_ylabel(
            "Amplitude"
        )

        axes[row, 0].grid(True)

        # 주파수 영역
        axes[row, 1].plot(
            frequencies,
            power,
            linewidth=0.7,
        )

        axes[row, 1].set_xlim(
            0,
            1000,
        )

        axes[row, 1].set_title(
            f"{bearing_code} spectrum"
        )

        axes[row, 1].set_xlabel(
            "Frequency (Hz)"
        )

        axes[row, 1].set_ylabel(
            "Power"
        )

        axes[row, 1].grid(True)

    figure.suptitle(
        "K / KA / KB comparison "
        f"({CONDITION})"
    )

    figure.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "compare_k_ka_kb.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)

    print()
    print(
        f"그래프 저장 완료: {output_path}"
    )


if __name__ == "__main__":
    main()
