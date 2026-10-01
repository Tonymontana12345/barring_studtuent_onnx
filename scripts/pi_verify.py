"""Raspberry Pi ONNX 수치 검증: Mac에서 계산한 logits와 같은지 비교한다.

필요 패키지: numpy, onnxruntime (bearing_kd 설치 불필요)
사용법:
    python pi_verify.py --model models/student.onnx --data pi_verify_data.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

WINDOW_SIZE, CHUNK_SIZE = 4000, 512
LABELS = {0: "Normal", 1: "Fault"}


def chunk_window(window):
    # bearing_kd.inference.chunk_window와 같은 규칙: 512×8 청크, 마지막 96개는 0 패딩 + mask 0
    count = (WINDOW_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunks = np.pad(window, (0, count * CHUNK_SIZE - WINDOW_SIZE)).reshape(count, 1, CHUNK_SIZE)
    mask = (np.arange(count * CHUNK_SIZE) < WINDOW_SIZE).astype(np.float32).reshape(count, CHUNK_SIZE)
    return chunks.astype(np.float32), mask


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=here / "student.onnx")
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
        actual.append(np.mean(chunk_logits, axis=0))  # 윈도우 판정 = 8개 청크 logits 평균
    actual = np.asarray(actual)

    diff = np.abs(actual - expected)
    same_pred = actual.argmax(1) == expected.argmax(1)
    correct = actual.argmax(1) == labels
    print(f"윈도우 {len(windows)}개")
    print(f"Mac 대비 logits 최대 차이: {diff.max():.3e}  (기준 1e-4 이하면 통과)")
    print(f"Mac과 예측 일치: {int(same_pred.sum())}/{len(windows)}")
    print(f"정답 라벨 일치: {int(correct.sum())}/{len(windows)}")
    print(f"청크당 평균 {np.mean(chunk_times) * 1000:.2f} ms, 윈도우당 {np.mean(chunk_times) * 8000:.1f} ms")
    print()
    files = sorted({n.split(":")[0] for n in names}, key=list(n.split(":")[0] for n in names).index)
    for stem in files:
        k = np.array([n.startswith(stem + ":") for n in names])
        preds = actual[k].argmax(1).tolist()
        print(f"{stem:22s} 정답={LABELS[int(labels[k][0])]:6s} 예측={preds}  최대차이={diff[k].max():.1e}")
    ok = diff.max() <= 1e-4 and same_pred.all()
    print("\n결과:", "PASS - Mac과 수치 일치" if ok else "FAIL - Mac과 결과가 다름")


if __name__ == "__main__":
    main()
