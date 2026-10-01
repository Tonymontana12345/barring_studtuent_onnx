"""Opt-in integration verification using local data/models, never replacing artifacts.

Run after an editable install: python scripts/verify_pipeline.py
Outputs are placed in a unique directory under outputs/verification/.
"""

import argparse
import gc
import json
import platform
import tempfile
from pathlib import Path

import numpy as np
import torch

from bearing_kd.config import PREPROCESS_CONFIG, PROJECT_ROOT
from bearing_kd.data import BearingWindowDataset, load_split_paths, make_moment_chunks
from bearing_kd.export import export_student
from bearing_kd.inference import create_session, predict_windows
from bearing_kd.models import load_model
from bearing_kd.preprocessing import bearing_label, preprocess_file
from bearing_kd.training import cache_teacher, evaluate, metrics, predict, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--skip-training", action="store_true", help="Only artifact parity/evaluation")
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(2)
    parent = root / "outputs/verification"
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    print(f"Verification output: {out}", flush=True)
    report = {"platform": platform.platform(), "python": platform.python_version()}
    small, base = root / "models/MOMENT-1-small", root / "models/MOMENT-1-base"
    student_ckpt = root / "outputs/checkpoints/student_kd_best.pt"
    teacher_ckpt = root / "outputs/checkpoints/teacher_lora_best.pt"
    split_path = root / "data/splits/file_split.json"
    dataset = BearingWindowDataset(load_split_paths(split_path, root)[2], root)
    student = load_model(small, student_ckpt)
    torch_logits = predict(student, dataset)
    report["existing_student"] = metrics(torch_logits, dataset.labels)
    session = create_session(root / "outputs/onnx/student_moment_chunk.onnx")
    onnx_logits = predict_windows(session, dataset.windows)
    np.testing.assert_allclose(onnx_logits, torch_logits, rtol=1e-4, atol=1e-5)
    report["existing_onnx"] = metrics(onnx_logits, dataset.labels)
    report["existing_onnx_max_error"] = float(np.max(np.abs(onnx_logits - torch_logits)))
    print("Existing Student and ONNX full test split verified", flush=True)
    del student, session
    gc.collect()
    report["existing_teacher"] = evaluate(base, teacher_ckpt, split_path, root, teacher=True)
    print("Existing Teacher full test split verified", flush=True)
    report["fresh_export"] = export_student(small, student_ckpt, out / "student.onnx", dataset.windows[0])
    del dataset
    gc.collect()
    # One real recording from every supported bearing family goes through the same filter.
    processed = out / "processed"
    raw_errors = {}
    for code in ("K001", "KA04", "KB24", "KI03"):
        raw = next(iter(sorted((root / "data/raw/paderborn" / code).glob("*.mat"))), None)
        if raw is None:
            raise FileNotFoundError(f"Missing raw data for integration check: {code}")
        windows = preprocess_file(raw)
        if code in ("K001", "KA04"):
            old = root / "data/processed/windows_1khz" / code / f"{raw.stem}_windows.npy"
            np.testing.assert_allclose(windows, np.load(old), rtol=1e-5, atol=1e-7)
            raw_errors[code] = float(np.max(np.abs(windows - np.load(old))))
        report.setdefault("raw_preprocessing", {})[code] = {"windows": len(windows), "label": bearing_label(code)}
    report["legacy_k_ka_preprocess_max_error"] = raw_errors
    if not args.skip_training:
        # Distinct recordings across splits, one window each; this checks mechanics, not accuracy.
        mini_split = {name: [] for name in ("train", "validation", "test")}
        for code in ("K001", "KA04"):
            raw_paths = sorted((root / "data/raw/paderborn" / code).glob("*.mat"))[:3]
            if len(raw_paths) != 3:
                raise ValueError("Smoke training needs three recordings per class")
            folder = processed / code
            folder.mkdir(parents=True, exist_ok=True)
            for name, raw in zip(mini_split, raw_paths):
                windows = preprocess_file(raw)[:1]
                path = folder / f"{raw.stem}_windows.npy"
                np.save(path, windows)
                np.save(folder / f"{raw.stem}_labels.npy", np.array([bearing_label(code)], dtype=np.int64))
                mini_split[name].append(str(path.relative_to(root)))
        (processed / "preprocessing.json").write_text(json.dumps(PREPROCESS_CONFIG, indent=2))
        mini_path = out / "split.json"
        mini_path.write_text(json.dumps(mini_split, indent=2))
        teacher_dir, student_dir, cache_dir = out / "teacher", out / "student", out / "cache"
        report["teacher_training_smoke"] = train(base, mini_path, teacher_dir, root, teacher=True, epochs=1, batch_size=2)
        trained = torch.load(teacher_dir / "teacher_lora_best.pt", map_location="cpu", weights_only=True)
        lora_b = [p for name, p in trained["model_state_dict"].items() if "lora_B" in name]
        if not lora_b or not all(torch.count_nonzero(p).item() > 0 for p in lora_b):
            raise AssertionError("Teacher LoRA B matrices did not update from their zero initialization")
        report["teacher_training_smoke"]["updated_lora_b_matrices"] = len(lora_b)
        del trained
        report["cache_smoke"] = cache_teacher(base, teacher_dir / "teacher_lora_best.pt", mini_path, cache_dir, root, batch_size=2)
        report["student_training_smoke"] = train(small, mini_path, student_dir, root, cache_dir=cache_dir, epochs=1, batch_size=2)
        report["student_smoke_evaluation"] = evaluate(small, student_dir / "student_kd_best.pt", mini_path, root, batch_size=2)
        print("Teacher train → cache → Student KD → strict reload/evaluation verified", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
