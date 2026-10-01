"""Validated datasets, file splits, and sample identities for distillation."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import CHUNK_SIZE, PROJECT_ROOT, WINDOW_SIZE, resolve_path
from .preprocessing import bearing_label


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BearingWindowDataset(Dataset):
    def __init__(self, windows_paths, root=PROJECT_ROOT):
        windows_list, labels_list, sample_ids = [], [], []
        digest = hashlib.sha256()
        self.paths = [resolve_path(p, root) for p in windows_paths]
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("Duplicate window files in dataset")
        for path in self.paths:
            if not path.name.endswith("_windows.npy"):
                raise ValueError(f"Not a window file: {path}")
            label_path = path.with_name(path.name.replace("_windows.npy", "_labels.npy"))
            windows = np.load(path, allow_pickle=False)
            labels = np.load(label_path, allow_pickle=False)
            if windows.ndim != 2 or windows.shape[1] != WINDOW_SIZE or len(windows) == 0:
                raise ValueError(f"Invalid window shape {windows.shape}: {path}")
            if labels.shape != (len(windows),) or not np.issubdtype(labels.dtype, np.integer):
                raise ValueError(f"Invalid label shape/dtype: {label_path}")
            if not np.all(labels == bearing_label(path.parent.name)):
                raise ValueError(f"Labels disagree with bearing code: {label_path}")
            if not np.isfinite(windows).all():
                raise ValueError(f"Non-finite windows: {path}")
            identity = f"{path.parent.name}/{path.name}"
            sample_ids.extend(f"{identity}:{i}" for i in range(len(windows)))
            digest.update(identity.encode())
            digest.update(sha256_file(path).encode())
            digest.update(sha256_file(label_path).encode())
            manifest = path.parent.parent / "preprocessing.json"
            digest.update(sha256_file(manifest).encode() if manifest.exists() else b"legacy-no-manifest")
            windows_list.append(windows.astype(np.float32))
            labels_list.append(labels.astype(np.int64))
        if not windows_list:
            raise ValueError("Dataset has no window files")
        self.windows = np.concatenate(windows_list)
        self.labels = np.concatenate(labels_list)
        self.sample_ids = np.asarray(sample_ids)
        self.fingerprint = digest.hexdigest()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return torch.from_numpy(self.windows[index]).unsqueeze(0), torch.tensor(self.labels[index])


def load_split_paths(split_path, root=PROJECT_ROOT):
    split = json.loads(resolve_path(split_path, root).read_text())
    groups = []
    seen = set()
    for name in ("train", "validation", "test"):
        paths = [resolve_path(p, root) for p in split[name]]
        if not paths or len(set(paths)) != len(paths) or seen.intersection(paths):
            raise ValueError(f"Empty, duplicate or overlapping split: {name}")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        seen.update(paths)
        groups.append(paths)
    return tuple(groups)


def create_split(processed_dir, output_path, root=PROJECT_ROOT, bearings=None, seed=42, test_bearings=None):
    """Stratify by binary label, grouping all windows from one recording together.

    Without test_bearings this is a recording-level 70/15/15 split, not a held-out-bearing
    evaluation. With test_bearings, every recording of those bearings is test data and the
    remaining bearings are split 85/15 into train/validation by recording.
    """
    paths = sorted(Path(processed_dir).rglob("*_windows.npy"))
    if bearings:
        for code in bearings:
            bearing_label(code)
        paths = [p for p in paths if p.parent.name in bearings]
        missing = set(bearings) - {p.parent.name for p in paths}
        if missing:
            raise FileNotFoundError(f"No processed files for: {sorted(missing)}")
    # Check labels and preprocessing files before emitting a usable split.
    BearingWindowDataset(paths, root)
    rng = random.Random(seed)
    groups = {"train": [], "validation": [], "test": []}
    # [추가: 베어링 단위 holdout]
    # 기존 split은 같은 베어링의 녹음 파일을 train/test로 나눠서, 99.6%가 나와도
    # 새 베어링에 대한 성능은 알 수 없었다(실제로 KB24는 0/560 정답).
    # test_bearings로 지정한 베어링은 파일 전체를 test로 보내 학습에서 완전히 제외한다.
    if test_bearings:
        test_bearings = set(test_bearings)
        missing = test_bearings - {p.parent.name for p in paths}
        if missing:
            raise FileNotFoundError(f"No processed files for test bearings: {sorted(missing)}")
        groups["test"] = [p for p in paths if p.parent.name in test_bearings]
        paths = [p for p in paths if p.parent.name not in test_bearings]
        # 고장 베어링만 test에 두면 "항상 고장" 모델도 100%가 나온다. 정상 오판(오경보)을
        # 측정할 수 있도록 정상/고장 베어링이 모두 있어야 한다.
        if {bearing_label(p.parent.name) for p in groups["test"]} != {0, 1}:
            raise ValueError("Held-out test bearings must include both normal and fault bearings")
    for label in (0, 1):
        subset = [p for p in paths if bearing_label(p.parent.name) == label]
        if len(subset) < 7:
            raise ValueError("At least 7 recordings per class are needed for splitting")
        rng.shuffle(subset)
        if test_bearings:
            a = int(len(subset) * .85)
            parts = (subset[:a], subset[a:], [])
        else:
            a, b = int(len(subset) * .7), int(len(subset) * .15)
            parts = (subset[:a], subset[a:a+b], subset[a+b:])
        for name, selected in zip(groups, parts):
            groups[name].extend(selected)
    for name, selected in groups.items():
        rng.shuffle(selected)
        # Paths are rooted at --root, never the caller's current directory.
        groups[name] = [str(p.resolve().relative_to(Path(root).resolve())) for p in selected]
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace an existing split: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(groups, indent=2) + "\n")
    return {name: len(paths) for name, paths in groups.items()}


def make_moment_chunks(batch_x, batch_y=None, moment_length=CHUNK_SIZE):
    if batch_x.ndim != 3 or batch_x.shape[1] != 1 or batch_x.shape[-1] < 1:
        raise ValueError("Expected [batch, 1, signal_length] input")
    if moment_length < 1:
        raise ValueError("Chunk length must be positive")
    batch, channels, length = batch_x.shape
    count = (length + moment_length - 1) // moment_length
    padded = F.pad(batch_x, (0, count * moment_length - length))
    chunks = padded.reshape(batch, channels, count, moment_length)
    chunks = chunks.permute(0, 2, 1, 3).reshape(batch * count, channels, moment_length)
    mask = torch.arange(count * moment_length, device=batch_x.device) < length
    mask = mask.to(batch_x.dtype).reshape(1, count, moment_length).expand(batch, -1, -1)
    labels = None if batch_y is None else batch_y.repeat_interleave(count)
    return chunks, mask.reshape(batch * count, moment_length), labels


class DistillationDataset(Dataset):
    def __init__(self, windows_paths, cache_path, root=PROJECT_ROOT):
        self.base_dataset = BearingWindowDataset(windows_paths, root)
        with np.load(cache_path, allow_pickle=False) as cache:
            required = {"logits", "labels", "sample_ids", "dataset_fingerprint", "teacher_sha256"}
            if not required.issubset(cache.files):
                raise ValueError("Legacy/incomplete teacher cache. Regenerate with cache-teacher.")
            if str(cache["dataset_fingerprint"].item()) != self.base_dataset.fingerprint:
                raise ValueError("Teacher cache dataset fingerprint mismatch; regenerate cache")
            if not np.array_equal(cache["sample_ids"], self.base_dataset.sample_ids):
                raise ValueError("Teacher cache sample order mismatch")
            if not np.array_equal(cache["labels"], self.base_dataset.labels):
                raise ValueError("Teacher cache labels mismatch")
            logits = cache["logits"]
            if logits.shape != (len(self.base_dataset), 2) or not np.isfinite(logits).all():
                raise ValueError("Invalid teacher logits")
            self.teacher_sha256 = str(cache["teacher_sha256"].item())
            self.teacher_logits = torch.from_numpy(logits.copy()).float()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        return x, y, self.teacher_logits[index]
