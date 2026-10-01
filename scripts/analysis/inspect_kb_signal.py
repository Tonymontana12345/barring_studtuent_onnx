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


def main():
    kb_paths = sorted(
        DATA_DIR.glob(
            "KB*/*_windows.npy"
        )
    )

    if not kb_paths:
        raise FileNotFoundError(
            "KB Window 파일을 찾지 못했습니다."
        )

    kb_path = kb_paths[0]

    windows = np.load(kb_path)

    # 첫 번째 KB Window 선택
    signal = windows[0]

    print("[KB 신호 확인]")
    print(f"파일: {kb_path}")
    print(f"전체 Window shape: {windows.shape}")
    print(f"선택한 Window shape: {signal.shape}")
    print(f"min : {signal.min():.6f}")
    print(f"max : {signal.max():.6f}")
    print(f"mean: {signal.mean():.6f}")
    print(f"std : {signal.std():.6f}")
    print(
        f"rms : "
        f"{np.sqrt(np.mean(signal ** 2)):.6f}"
    )

    frequencies, power = periodogram(
        signal,
        fs=FS,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    time_axis = np.arange(
        len(signal)
    ) / FS

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
    )

    axes[0].plot(
        time_axis,
        signal,
        linewidth=0.8,
    )

    axes[0].set_title(
        "KB vibration waveform"
    )
    axes[0].set_xlabel(
        "Time (seconds)"
    )
    axes[0].set_ylabel(
        "Amplitude"
    )
    axes[0].grid(True)

    axes[1].plot(
        frequencies,
        power,
        linewidth=0.8,
    )

    axes[1].set_xlim(
        0,
        1000,
    )

    axes[1].set_title(
        "KB vibration spectrum"
    )
    axes[1].set_xlabel(
        "Frequency (Hz)"
    )
    axes[1].set_ylabel(
        "Power"
    )
    axes[1].grid(True)

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "kb_signal_inspection.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)

    print()
    print(f"그래프 저장: {output_path}")


if __name__ == "__main__":
    main()
