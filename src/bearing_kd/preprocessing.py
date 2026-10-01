"""One preprocessing implementation for K, KA, KB and KI bearings."""

import json
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

from .config import (
    CUTOFF_HZ, FILTER_ORDER, PREPROCESS_CONFIG, RAW_FS, SAMPLE_RATE,
    STRIDE, WINDOW_SIZE,
)


def bearing_label(code):
    if re.fullmatch(r"K\d{3}", code):
        return 0
    if re.fullmatch(r"K[ABI]\d{2}", code):
        return 1
    raise ValueError(f"Unknown bearing code: {code}")


def extract_vibration(path):
    path = Path(path)
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if path.stem not in mat or not hasattr(mat[path.stem], "Y"):
        raise ValueError(f"Missing measurement record/Y in {path}")
    for item in np.atleast_1d(mat[path.stem].Y):
        name = item.Name
        if isinstance(name, np.ndarray) and name.size == 1:
            name = name.item()
        if str(name) == "vibration_1":
            signal = np.asarray(item.Data, dtype=np.float32).reshape(-1)
            if signal.size == 0 or not np.isfinite(signal).all():
                raise ValueError(f"Empty or non-finite vibration signal: {path}")
            return signal
    raise ValueError(f"vibration_1 not found: {path}")


def preprocess_signal(signal):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or signal.size < RAW_FS:
        raise ValueError("Expected at least one second of a 1-D 64 kHz signal")
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains NaN or infinity")
    sos = butter(FILTER_ORDER, CUTOFF_HZ / (RAW_FS / 2), btype="lowpass", output="sos")
    # Preserve the float32 intermediate used for the original K/KA training data.
    filtered = sosfiltfilt(sos, signal).astype(np.float32)
    return resample_poly(filtered, up=1, down=RAW_FS // SAMPLE_RATE).astype(np.float32)


def make_windows(signal):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or len(signal) < WINDOW_SIZE or not np.isfinite(signal).all():
        raise ValueError("Expected a finite 1-D signal with at least 4000 samples")
    return np.stack([
        signal[start:start + WINDOW_SIZE]
        for start in range(0, len(signal) - WINDOW_SIZE + 1, STRIDE)
    ])


def preprocess_file(path):
    return make_windows(preprocess_signal(extract_vibration(path)))


def preprocess_directory(raw_dir, output_dir, bearings=None):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    paths = sorted(raw_dir.rglob("*.mat"))
    if bearings:
        for code in bearings:
            bearing_label(code)
        paths = [p for p in paths if p.parent.name in bearings]
        missing = set(bearings) - {p.parent.name for p in paths}
        if missing:
            raise FileNotFoundError(f"No MAT files for: {sorted(missing)}")
    if not paths:
        raise FileNotFoundError(f"No MAT files in {raw_dir}")
    # Never mix new preprocessing with an existing dataset by accident.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new, empty output directory: {output_dir}")
    for path in paths:
        bearing_label(path.parent.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    # An exception stops the run; manifest is written only after all files succeed.
    for index, path in enumerate(paths, 1):
        windows = preprocess_file(path)
        labels = np.full(len(windows), bearing_label(path.parent.name), dtype=np.int64)
        folder = output_dir / path.parent.name
        folder.mkdir(exist_ok=True)
        np.save(folder / f"{path.stem}_windows.npy", windows)
        np.save(folder / f"{path.stem}_labels.npy", labels)
        total += len(windows)
        print(f"[{index}/{len(paths)}] {path.name}: {len(windows)} windows", flush=True)
    manifest = {**PREPROCESS_CONFIG, "files": len(paths), "windows": total}
    (output_dir / "preprocessing.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
