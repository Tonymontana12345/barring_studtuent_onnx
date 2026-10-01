"""Shared preprocessing contract and portable project paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FS = 64_000
SAMPLE_RATE = 4_000
CUTOFF_HZ = 1_000
FILTER_ORDER = 8
WINDOW_SIZE = 4_000
STRIDE = 2_000
CHUNK_SIZE = 512
PREPROCESS_VERSION = "v2-butter8-float32-4khz"
PREPROCESS_CONFIG = {
    "version": PREPROCESS_VERSION,
    "raw_sample_rate": RAW_FS,
    "sample_rate": SAMPLE_RATE,
    "cutoff_hz": CUTOFF_HZ,
    "filter_order": FILTER_ORDER,
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "chunk_size": CHUNK_SIZE,
    "filter": "butter-sosfiltfilt",
}


def resolve_path(path, root=PROJECT_ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()
