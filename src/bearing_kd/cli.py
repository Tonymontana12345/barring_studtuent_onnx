"""Command-line entry point. Mutating commands require explicit output locations."""

import argparse
import json
from pathlib import Path

from .config import PROJECT_ROOT, resolve_path


def parser():
    p = argparse.ArgumentParser(description="Paderborn → MOMENT LoRA → KD → ONNX")
    p.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Base directory for data/models/outputs paths")
    p.add_argument("--threads", type=int, default=2, help="CPU inference/training threads")
    commands = p.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preprocess", help="Unified 8th-order preprocessing into a new directory")
    pre.add_argument("--raw-dir", default="data/raw/paderborn")
    pre.add_argument("--output-dir", required=True)
    pre.add_argument("--bearings", nargs="+")
    split = commands.add_parser("split", help="Recording-level 70/15/15 split, or held-out test bearings")
    split.add_argument("--processed-dir", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--bearings", nargs="+")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--test-bearings", nargs="+", help="Hold out these bearings entirely as test data")
    for name in ("train-teacher", "train-student", "cache-teacher", "evaluate-teacher", "evaluate-student"):
        sub = commands.add_parser(name)
        teacher = name.endswith("teacher")
        sub.add_argument("--model-dir", default=f"models/MOMENT-1-{'base' if teacher else 'small'}")
        sub.add_argument("--split", default="data/splits/file_split.json")
        # mps: Apple Silicon GPU(Metal). LoRA 수정 후 Teacher 역전파가 인코더 전체를 통과해
        # CPU로는 20 epoch에 수 시간이 걸려 추가했다. 측정 결과 CPU 대비 약 2배 빠르다.
        sub.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
        sub.add_argument("--batch-size", type=int, default=16)
        if name.startswith("train"):
            sub.add_argument("--output-dir", required=True)
            sub.add_argument("--epochs", type=int, default=20)
            sub.add_argument("--learning-rate", type=float, default=1e-4)
            sub.add_argument("--seed", type=int, default=42)
            if not teacher:
                sub.add_argument("--cache-dir", required=True)
        else:
            sub.add_argument("--checkpoint", default=f"outputs/checkpoints/{'teacher_lora' if teacher else 'student_kd'}_best.pt")
            if name == "cache-teacher":
                sub.add_argument("--output-dir", required=True)
            else:
                sub.add_argument("--processed-dir", help="Evaluate all windows here instead of test split (e.g. KB24)")
    export = commands.add_parser("export", help="Export ONNX and assert numerical parity")
    export.add_argument("--model-dir", default="models/MOMENT-1-small")
    export.add_argument("--checkpoint", default="outputs/checkpoints/student_kd_best.pt")
    export.add_argument("--output", required=True)
    export.add_argument("--sample", help="Optional preprocessed NPY windows for real-signal parity")
    infer = commands.add_parser("infer", help="Full-window inference on preprocessed NPY or raw 64 kHz MAT")
    infer.add_argument("--model", default="outputs/onnx/student_moment_chunk.onnx")
    infer.add_argument("--input", required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.threads < 1 or getattr(args, "batch_size", 1) < 1:
        p.error("threads and batch-size must be positive")
    root = args.root.resolve()
    path = lambda value: resolve_path(value, root)
    command = args.command
    if command == "preprocess":
        from .preprocessing import preprocess_directory
        result = preprocess_directory(path(args.raw_dir), path(args.output_dir), args.bearings)
    elif command == "split":
        from .data import create_split
        result = create_split(path(args.processed_dir), path(args.output), root, args.bearings, args.seed,
                              args.test_bearings)
    elif command == "infer":
        import numpy as np
        from .inference import create_session, predict_windows
        from .preprocessing import preprocess_file
        input_path = path(args.input)
        if input_path.suffix == ".mat":
            windows = preprocess_file(input_path)
        elif input_path.suffix == ".npy":
            windows = np.load(input_path, allow_pickle=False)
        else:
            p.error("--input must be a raw .mat recording or preprocessed .npy windows")
        logits = predict_windows(create_session(path(args.model), args.threads), windows)
        result = {"labels": logits.argmax(1).tolist(), "logits": logits.tolist(), "label_map": {0: "Normal", 1: "Fault"}}
    else:
        import torch
        torch.set_num_threads(args.threads)
        if getattr(args, "device", "cpu") == "cuda" and not torch.cuda.is_available():
            p.error("CUDA requested but unavailable")
        if getattr(args, "device", "cpu") == "mps" and not torch.backends.mps.is_available():
            p.error("MPS requested but unavailable")
        if command == "export":
            import numpy as np
            from .export import export_student
            sample = None
            if args.sample:
                sample = np.load(path(args.sample), allow_pickle=False)
                if sample.ndim == 2:
                    sample = sample[0]
            result = export_student(path(args.model_dir), path(args.checkpoint), path(args.output), sample)
        else:
            from .training import cache_teacher, evaluate, train
            common = dict(model_dir=path(args.model_dir), split_path=path(args.split), root=root,
                          device=args.device, batch_size=args.batch_size)
            if command.startswith("train"):
                result = train(**common, output_dir=path(args.output_dir), teacher=command == "train-teacher",
                               cache_dir=path(args.cache_dir) if command == "train-student" else None,
                               epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed)
            elif command == "cache-teacher":
                result = cache_teacher(**common, checkpoint=path(args.checkpoint), output_dir=path(args.output_dir))
            else:
                result = evaluate(**common, checkpoint=path(args.checkpoint), teacher=command == "evaluate-teacher",
                                  processed_dir=path(args.processed_dir) if args.processed_dir else None)
    print(json.dumps(result, indent=2, ensure_ascii=False))
