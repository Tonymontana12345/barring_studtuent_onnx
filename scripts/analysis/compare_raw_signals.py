from bearing_kd.config import PROJECT_ROOT

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import welch
FS = 64000
DATA_DIR = PROJECT_ROOT / 'data/raw/paderborn'
PLOT_DIR = PROJECT_ROOT / 'outputs/plots'

def extract_vibration(path):
    print(path)
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = path.stem
    record = mat[key]
    for item in record.Y:
        name = item.Name
        if isinstance(name, np.ndarray) and name.size == 1:
            name = name.item()
        if str(name) == 'vibration_1':
            return np.asarray(item.Data).reshape(-1).astype(np.float32)
    raise RuntimeError(f'vibration_1을 찾지 못했습니다: {path}')

def print_summary(name, signal):
    print(f'\n[{name}]')
    print('samples:', len(signal))
    print('mean:', signal.mean())
    print('std:', signal.std())
    print('rms:', np.sqrt(np.mean(signal ** 2)))
    print('min:', signal.min())
    print('max:', signal.max())

def analyze_signal(name, siganl):
    siganl = siganl - siganl.mean()
    rms = np.sqrt(np.mean(siganl ** 2))
    std = np.std(siganl)
    peak = np.max(np.abs(siganl))
    crest_factor = peak / (rms + 1e-08)
    freq, power = welch(siganl, fs=FS, nperseg=min(16384, len(siganl)))
    low_band = freq <= 1000
    total_band = freq <= 5000
    low_energy = np.sum(power[low_band])
    total_energy = np.sum(power[total_band])
    low_energy_ratio = low_energy / (total_energy + 1e-12)
    dominant_index = np.argmax(power[1:]) + 1
    dominant_frequency = freq[dominant_index]
    print(f'\n[{name}]')
    print(f'RMS: {rms:.6f}')
    print(f'STD: {std:.6f}')
    print(f'Peak: {peak:.6f}')
    print(f'Crest factor: {crest_factor:.6f}')
    print(f'Dominant frequency: {dominant_frequency:.2f} Hz')
    print(f'Energy ratio below 1 kHz: {low_energy_ratio:.6f}')

def make_fft(signal):
    signal = signal - signal.mean()
    frequency = np.fft.rfftfreq(len(signal), 1 / FS)
    magnitude = np.abs(np.fft.rfft(signal))
    return (frequency, magnitude)
from scipy.signal import welch

def low_energy_ratio(signal):
    signal = signal - signal.mean()
    freq, power = welch(signal, fs=FS, nperseg=min(16384, len(signal)))
    low_band = freq <= 1000
    total_band = freq <= 5000
    low_energy = np.sum(power[low_band])
    total_energy = np.sum(power[total_band])
    return low_energy / (total_energy + 1e-12)

def compare_bearing_files(bearing_name):
    bearing_dir = DATA_DIR / bearing_name
    files = sorted(bearing_dir.glob(f'N09_M07_F10_{bearing_name}_*.mat'))
    ratios = []
    for path in files:
        signal = extract_vibration(path)
        ratio = low_energy_ratio(signal)
        ratios.append(ratio)
        print(f'{bearing_name} | {path.name} | ratio: {ratio:.6f}')
    ratios = np.asarray(ratios)
    print(f'\n[{bearing_name} summary]')
    print('count:', len(ratios))
    print('mean:', ratios.mean())
    print('std:', ratios.std())
    print('min:', ratios.min())
    print('max:', ratios.max())
    return ratios

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    healthy = extract_vibration(DATA_DIR / 'K001/N09_M07_F10_K001_1.mat')
    damaged = extract_vibration(DATA_DIR / 'KA04/N09_M07_F10_KA04_1.mat')
    n = min(len(healthy), len(damaged))
    healthy_cmp = healthy[:n]
    damaged_cmp = damaged[:n]
    analyze_signal('Healthy K001', healthy_cmp)
    analyze_signal('Damaged KA04', damaged_cmp)
    print_summary('Healthy K001', healthy)
    print_summary('Damaged KA04', damaged)
    sample_count = min(FS // 20, len(healthy), len(damaged))
    time = np.arange(sample_count) / FS
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(time, healthy[:sample_count])
    axes[0].set_title('Healthy K001')
    axes[0].set_ylabel('Raw amplitude')
    axes[0].grid()
    axes[1].plot(time, damaged[:sample_count])
    axes[1].set_title('Damaged KA04')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Raw amplitude')
    axes[1].grid()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'healthy_vs_damaged_waveform.png', dpi=150)
    plt.close(fig)
    freq_h, fft_h = make_fft(healthy)
    freq_d, fft_d = make_fft(damaged)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(freq_h, fft_h)
    axes[0].set_xlim(0, 5000)
    axes[0].set_title('Healthy K001 FFT')
    axes[0].set_ylabel('Magnitude')
    axes[0].grid()
    axes[1].plot(freq_d, fft_d)
    axes[1].set_xlim(0, 5000)
    axes[1].set_title('Damaged KA04 FFT')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].grid()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / 'healthy_vs_damaged_fft.png', dpi=150)
    plt.close(fig)
    healthy_ratios = compare_bearing_files('K001')
    damaged_ratios = compare_bearing_files('KA04')
    print('\nDifference')
    print('Damaged - Healthy:', damaged_ratios.mean() - healthy_ratios.mean())


if __name__ == "__main__":
    main()
