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

BANDS = [
    (0, 100),
    (100, 300),
    (300, 500),
    (500, 800),
    (800, 1000),
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
            f"{bearing_code} 파일을 찾지 못했습니다."
        )

    return paths[0]


def calculate_average_spectrum(
    windows,
):
    all_power = []

    for signal in windows:
        frequencies, power = periodogram(
            signal,
            fs=FS,
        )

        all_power.append(power)

    average_power = np.mean(
        all_power,
        axis=0,
    )

    return frequencies, average_power


def calculate_band_ratios(
    frequencies,
    power,
):
    total_energy = power.sum()

    results = []

    for low_hz, high_hz in BANDS:
        band_mask = (
            (frequencies >= low_hz)
            & (frequencies < high_hz)
        )

        band_energy = power[
            band_mask
        ].sum()

        ratio = band_energy / (
            total_energy + 1e-12
        )

        results.append(
            (
                low_hz,
                high_hz,
                band_energy,
                ratio,
            )
        )

    return results


def main():
    spectra = {}

    print("[주파수 비교]")

    for bearing_code in BEARING_CODES:
        path = find_window_file(
            bearing_code
        )

        windows = np.load(path)

        frequencies, power = (
            calculate_average_spectrum(
                windows
            )
        )

        spectra[bearing_code] = (
            frequencies,
            power,
        )

        peak_index = np.argmax(
            power[1:]
        ) + 1

        peak_frequency = frequencies[
            peak_index
        ]

        band_results = (
            calculate_band_ratios(
                frequencies,
                power,
            )
        )

        print()
        print(f"[{bearing_code}]")
        print(f"파일: {path}")
        print(
            f"Window shape: "
            f"{windows.shape}"
        )
        print(
            f"최대 에너지 주파수: "
            f"{peak_frequency:.2f} Hz"
        )

        print("대역별 에너지 비율:")

        for (
            low_hz,
            high_hz,
            band_energy,
            ratio,
        ) in band_results:
            print(
                f"{low_hz:>4}~{high_hz:<4} Hz: "
                f"{ratio:.6f}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 9),
    )

    # 평균 주파수 스펙트럼
    for bearing_code in BEARING_CODES:
        frequencies, power = spectra[
            bearing_code
        ]

        axes[0].plot(
            frequencies,
            power + 1e-12,
            label=bearing_code,
        )

    axes[0].set_xlim(
        0,
        1000,
    )

    axes[0].set_yscale(
        "log"
    )

    axes[0].set_title(
        "Average spectrum comparison"
    )

    axes[0].set_xlabel(
        "Frequency (Hz)"
    )

    axes[0].set_ylabel(
        "Power"
    )

    axes[0].legend()
    axes[0].grid(True)

    # 대역별 에너지 비율
    band_labels = [
        f"{low}-{high}"
        for low, high in BANDS
    ]

    x = np.arange(
        len(BANDS)
    )

    width = 0.25

    for index, bearing_code in enumerate(
        BEARING_CODES
    ):
        frequencies, power = spectra[
            bearing_code
        ]

        band_results = (
            calculate_band_ratios(
                frequencies,
                power,
            )
        )

        ratios = [
            result[3]
            for result in band_results
        ]

        axes[1].bar(
            x + index * width,
            ratios,
            width,
            label=bearing_code,
        )

    axes[1].set_xticks(
        x + width
    )

    axes[1].set_xticklabels(
        band_labels
    )

    axes[1].set_title(
        "Band energy ratio comparison"
    )

    axes[1].set_xlabel(
        "Frequency band (Hz)"
    )

    axes[1].set_ylabel(
        "Energy ratio"
    )

    axes[1].legend()
    axes[1].grid(
        axis="y"
    )

    figure.suptitle(
        "K / KA / KB spectrum comparison"
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "spectrum_k_ka_kb.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)

    print()
    print(
        f"그래프 저장: {output_path}"
    )


if __name__ == "__main__":
    main()
