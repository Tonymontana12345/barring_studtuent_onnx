"""ONNX parity check: compare logits against reference values computed on macOS.

Requires only numpy and onnxruntime.
Usage (from the repository root):
    python verify/pi_verify.py
    python verify/pi_verify.py --model onnx/student.onnx --data verify/pi_verify_data.npz --threads 4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

WINDOW_SIZE, CHUNK_SIZE = 4000, 512
LABELS = {0: "Normal", 1: "Fault"}


def chunk_window(window):
    # Same rule as bearing_kd.inference.chunk_window: 8 chunks of 512, last 96 samples zero-padded with mask 0
    count = (WINDOW_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunks = np.pad(window, (0, count * CHUNK_SIZE - WINDOW_SIZE)).reshape(count, 1, CHUNK_SIZE)
    mask = (np.arange(count * CHUNK_SIZE) < WINDOW_SIZE).astype(np.float32).reshape(count, CHUNK_SIZE)
    return chunks.astype(np.float32), mask


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=here.parent / "onnx" / "student.onnx")
    parser.add_argument("--data", type=Path, default=here / "pi_verify_data.npz")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    data = np.load(args.data, allow_pickle=False)
    windows, labels = data["windows"], data["labels"]
    expected, names = data["expected_logits"], data["names"]

    actual, chunk_times = [], []
    for window in windows:
        chunks, masks = chunk_window(window)
        chunk_logits = []
        for i in range(len(chunks)):
            start = time.perf_counter()
            out = session.run(["logits"], {"x_enc": chunks[i:i + 1], "input_mask": masks[i:i + 1]})[0][0]
            chunk_times.append(time.perf_counter() - start)
            chunk_logits.append(out)
        actual.append(np.mean(chunk_logits, axis=0))  # window decision = mean of the 8 chunk logits
    actual = np.asarray(actual)

    diff = np.abs(actual - expected)
    same_pred = actual.argmax(1) == expected.argmax(1)
    correct = actual.argmax(1) == labels
    print(f"windows: {len(windows)}")
    print(f"max |logit difference| vs reference: {diff.max():.3e}  (pass if <= 1e-4)")
    print(f"predictions matching reference: {int(same_pred.sum())}/{len(windows)}")
    print(f"correct vs true labels: {int(correct.sum())}/{len(windows)}")
    print(f"mean {np.mean(chunk_times) * 1000:.2f} ms per chunk, {np.mean(chunk_times) * 8000:.1f} ms per window")
    print()
    files = sorted({n.split(":")[0] for n in names}, key=list(n.split(":")[0] for n in names).index)
    for stem in files:
        k = np.array([n.startswith(stem + ":") for n in names])
        preds = actual[k].argmax(1).tolist()
        print(f"{stem:22s} true={LABELS[int(labels[k][0])]:6s} pred={preds}  max_diff={diff[k].max():.1e}")
    ok = diff.max() <= 1e-4 and same_pred.all()
    print("\nResult:", "PASS - matches reference" if ok else "FAIL - differs from reference")


if __name__ == "__main__":
    main()
