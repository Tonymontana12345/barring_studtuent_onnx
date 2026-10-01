"""4장 완성 코드: 원본 64 kHz 진동을 모델 입력용 1초 윈도우로 전처리한다.

처리 순서 (src/bearing_kd/preprocessing.py와 같은 규칙):
    .mat → vibration_1 꺼내기 → 1 kHz 저역통과(Butterworth 8차, 양방향) → 4 kHz로 리샘플링
         → 4000샘플(1초) 윈도우, 2000샘플씩 이동 → *_windows.npy, *_labels.npy, preprocessing.json 저장

실행 (프로젝트 루트에서):
    # 맛보기: 베어링 4개 × 파일 8개 (수십 초)
    python docs/manual/code/ch04_preprocess.py
    # 전체: 파일 수 제한 없이
    python docs/manual/code/ch04_preprocess.py --bearings K001 K002 K003 K004 KA03 KA04 KI03 KB24 KB27 \
        --max-files 0 --output-dir outputs/manual/processed_full
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parents[3]

# ── 전처리 규칙(계약). 학습·추론·라즈베리파이 모두 이 값을 똑같이 써야 한다. ──
RAW_FS = 64_000       # 원본 샘플링 레이트
SAMPLE_RATE = 4_000   # 낮춘 샘플링 레이트 (라즈베리파이 속도 때문에 선택)
CUTOFF_HZ = 1_000     # 저역통과 차단 주파수 (4 kHz의 절반 2 kHz보다 충분히 낮게)
FILTER_ORDER = 8      # 필터 차수: 클수록 차단이 가파르다
WINDOW_SIZE = 4_000   # 1초 = 4000 샘플
STRIDE = 2_000        # 0.5초씩 이동 → 윈도우끼리 절반 겹침


def bearing_label(code):
    if re.fullmatch(r"K\d{3}", code):
        return 0
    if re.fullmatch(r"K[ABI]\d{2}", code):
        return 1
    raise ValueError(f"알 수 없는 베어링 코드: {code}")


def extract_vibration(path):
    path = Path(path)
    record = loadmat(path, squeeze_me=True, struct_as_record=False)[path.stem]
    for item in np.atleast_1d(record.Y):
        if str(item.Name) == "vibration_1":
            signal = np.asarray(item.Data, dtype=np.float32).reshape(-1)
            if signal.size == 0 or not np.isfinite(signal).all():
                raise ValueError(f"비어 있거나 NaN/무한대가 있는 신호: {path}")
            return signal
    raise ValueError(f"vibration_1 없음: {path}")


def preprocess_signal(signal):
    """64 kHz 신호 → 1 kHz 저역통과 → 4 kHz."""
    signal = np.asarray(signal, dtype=np.float32)
    sos = butter(FILTER_ORDER, CUTOFF_HZ / (RAW_FS / 2), btype="lowpass", output="sos")
    filtered = sosfiltfilt(sos, signal).astype(np.float32)  # 앞뒤로 두 번 → 위상(시간) 밀림 없음
    return resample_poly(filtered, up=1, down=RAW_FS // SAMPLE_RATE).astype(np.float32)  # 16개 중 1개


def make_windows(signal):
    """긴 신호를 4000샘플짜리 조각으로, 2000샘플씩 밀면서 자른다."""
    starts = range(0, len(signal) - WINDOW_SIZE + 1, STRIDE)
    return np.stack([signal[s:s + WINDOW_SIZE] for s in starts])


def preprocess_file(path):
    return make_windows(preprocess_signal(extract_vibration(path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/paderborn")
    parser.add_argument("--output-dir", default="outputs/manual/processed")
    parser.add_argument("--bearings", nargs="+", default=["K001", "K002", "KA03", "KA04"])
    parser.add_argument("--max-files", type=int, default=8, help="베어링당 파일 수 (0이면 전부)")
    parser.add_argument("--overwrite", action="store_true", help="출력 폴더가 있으면 지우고 다시 만든다")
    args = parser.parse_args()

    raw_dir, out_dir = ROOT / args.raw_dir, ROOT / args.output_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out_dir.relative_to(ROOT)} 가 비어 있지 않습니다. --overwrite 를 쓰거나 다른 경로를 주세요.")
        shutil.rmtree(out_dir)

    total_files = total_windows = 0
    for code in args.bearings:
        label = bearing_label(code)
        paths = sorted((raw_dir / code).glob("*.mat"))
        if args.max_files:
            # 운전 조건 4가지가 고르게 들어가도록 조건별로 번갈아 고른다
            by_condition = {}
            for p in paths:
                by_condition.setdefault(p.name[:11], []).append(p)
            paths = [p for group in zip(*by_condition.values()) for p in group][:args.max_files]
        if not paths:
            raise SystemExit(f"{code}: .mat 파일이 없습니다 (3장의 압축 풀기 참고)")
        (out_dir / code).mkdir(parents=True, exist_ok=True)
        count = 0
        for path in paths:
            windows = preprocess_file(path)
            np.save(out_dir / code / f"{path.stem}_windows.npy", windows)
            np.save(out_dir / code / f"{path.stem}_labels.npy", np.full(len(windows), label, dtype=np.int64))
            count += len(windows)
        print(f"{code} (라벨 {label}): 파일 {len(paths)}개 → 윈도우 {count}개")
        total_files += len(paths)
        total_windows += count

    manifest = {"raw_sample_rate": RAW_FS, "sample_rate": SAMPLE_RATE, "cutoff_hz": CUTOFF_HZ,
                "filter_order": FILTER_ORDER, "filter": "butter-sosfiltfilt", "window_size": WINDOW_SIZE,
                "stride": STRIDE, "files": total_files, "windows": total_windows}
    (out_dir / "preprocessing.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\n합계: 파일 {total_files}개, 윈도우 {total_windows}개 → {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
