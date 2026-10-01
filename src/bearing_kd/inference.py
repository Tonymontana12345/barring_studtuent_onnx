"""CPU ONNX inference; this module does not import PyTorch or MOMENT."""

import numpy as np
import onnxruntime as ort

from .config import CHUNK_SIZE, WINDOW_SIZE


def chunk_window(window):
    window = np.asarray(window, dtype=np.float32)
    if window.shape != (WINDOW_SIZE,) or not np.isfinite(window).all():
        raise ValueError("Expected one finite, preprocessed 4000-sample window")
    count = (WINDOW_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunks = np.pad(window, (0, count * CHUNK_SIZE - WINDOW_SIZE)).reshape(count, 1, CHUNK_SIZE)
    mask = (np.arange(count * CHUNK_SIZE) < WINDOW_SIZE).astype(np.float32).reshape(count, CHUNK_SIZE)
    return chunks, mask


def create_session(model_path, threads=2):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    expected = {"x_enc": [1, 1, CHUNK_SIZE], "input_mask": [1, CHUNK_SIZE]}
    if {i.name: i.shape for i in session.get_inputs()} != expected:
        raise ValueError("Expected the fixed-batch MOMENT chunk ONNX input contract")
    if any(i.type != "tensor(float)" for i in session.get_inputs()):
        raise ValueError("ONNX inputs must be float32")
    return session


def predict_windows(session, windows):
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim == 1:
        windows = windows[None, :]
    if windows.ndim != 2 or windows.shape[1] != WINDOW_SIZE or not len(windows):
        raise ValueError("Expected [N, 4000] preprocessed windows")
    results = []
    for window in windows:
        chunks, masks = chunk_window(window)
        logits = [session.run(["logits"], {"x_enc": chunks[i:i+1], "input_mask": masks[i:i+1]})[0][0]
                  for i in range(len(chunks))]
        results.append(np.mean(logits, axis=0))
    result = np.asarray(results)
    if result.shape != (len(windows), 2) or not np.isfinite(result).all():
        raise RuntimeError("Invalid ONNX model output")
    return result
