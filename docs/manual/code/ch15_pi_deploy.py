"""15장 완성 코드: 라즈베리파이로 가져갈 배포 묶음을 만들고, 실시간 처리 가능 여부를 계산한다.

Mac(학습 컴퓨터)에서 실행:
    python docs/manual/code/ch15_pi_deploy.py
→ outputs/manual/pi_bundle/ 에 다음을 만든다
    student.onnx, student.onnx.data   모델 (이름 바꾸지 말 것)
    pi_verify_data.npz                 전처리된 윈도우 + Mac에서 계산한 기준 logits
    pi_infer.py                        라즈베리파이용 추론·검증 스크립트 (numpy + onnxruntime만 사용)

라즈베리파이에서 실행:
    python pi_infer.py verify                  # Mac 결과와 같은지 + 속도
    python pi_infer.py mat 파일.mat            # 원본 .mat을 전처리해서 판정 (scipy 필요)
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

PI_SCRIPT = r'''"""Raspberry Pi: ONNX 진동 판정 + 검증. 필요 패키지: numpy, onnxruntime (mat 모드는 scipy도)."""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
WINDOW, CHUNK = 4000, 512


def session(threads=4):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(HERE / "student.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])


def predict_window(sess, window):
    chunks = np.pad(window.astype(np.float32), (0, 96)).reshape(8, 1, CHUNK)
    masks = (np.arange(8 * CHUNK) < WINDOW).astype(np.float32).reshape(8, CHUNK)
    logits = [sess.run(["logits"], {"x_enc": chunks[i:i + 1], "input_mask": masks[i:i + 1]})[0][0] for i in range(8)]
    return np.mean(logits, axis=0)                       # 청크 8개 logits 평균 = 윈도우 판정


def preprocess_mat(path):
    """학습과 똑같은 전처리 (4장): 1 kHz 저역통과(8차, 양방향) → 4 kHz → 1초 윈도우, 0.5초 이동."""
    from scipy.io import loadmat
    from scipy.signal import butter, resample_poly, sosfiltfilt
    path = Path(path)
    record = loadmat(path, squeeze_me=True, struct_as_record=False)[path.stem]
    x = next(np.asarray(i.Data, dtype=np.float32).ravel() for i in np.atleast_1d(record.Y) if str(i.Name) == "vibration_1")
    sos = butter(8, 1000 / 32000, btype="lowpass", output="sos")
    x = resample_poly(sosfiltfilt(sos, x).astype(np.float32), 1, 16).astype(np.float32)
    return np.stack([x[s:s + WINDOW] for s in range(0, len(x) - WINDOW + 1, 2000)])


def verify(sess):
    data = np.load(HERE / "pi_verify_data.npz")
    start = time.perf_counter()
    actual = np.array([predict_window(sess, w) for w in data["windows"]])
    per_window = (time.perf_counter() - start) / len(actual) * 1000
    diff = float(np.abs(actual - data["expected_logits"]).max())
    same = int((actual.argmax(1) == data["expected_logits"].argmax(1)).sum())
    print(f"윈도우 {len(actual)}개 | Mac 대비 최대 차이 {diff:.1e} | 예측 일치 {same}/{len(actual)}")
    print(f"윈도우당 {per_window:.0f} ms (실시간 한계 500 ms) → {'실시간 가능' if per_window < 500 else '실시간 불가'}")
    print("결과:", "PASS" if diff <= 1e-4 and same == len(actual) else "FAIL")


if __name__ == "__main__":
    sess = session()
    if len(sys.argv) >= 3 and sys.argv[1] == "mat":
        for i, w in enumerate(preprocess_mat(sys.argv[2])):
            logit = predict_window(sess, w)
            print(f"윈도우 {i}: logits [{logit[0]:+.3f}, {logit[1]:+.3f}] → {'고장' if logit.argmax() else '정상'}")
    else:
        verify(sess)
'''


def predict_window(sess, window):
    """PI_SCRIPT의 predict_window와 같은 계산: 청크 8개 → logits 평균."""
    chunks = np.pad(window.astype(np.float32), (0, 96)).reshape(8, 1, 512)
    masks = (np.arange(8 * 512) < 4000).astype(np.float32).reshape(8, 512)
    return np.mean([sess.run(["logits"], {"x_enc": chunks[i:i + 1], "input_mask": masks[i:i + 1]})[0][0]
                    for i in range(8)], axis=0)


def make_bundle(onnx_path, windows_files, out_dir):
    from ch06_dataset import WindowDataset
    import onnxruntime as ort

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in (onnx_path, Path(str(onnx_path) + ".data")):
        shutil.copy2(f, out_dir / f.name)                 # .onnx와 .onnx.data는 같은 폴더, 같은 이름
    (out_dir / "pi_infer.py").write_text(PI_SCRIPT)

    ds = WindowDataset(windows_files)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    start = time.perf_counter()
    expected = [predict_window(sess, w) for w in ds.windows]   # 라즈베리파이와 똑같은 방식으로 기준값 계산
    mac_ms = (time.perf_counter() - start) / len(ds) * 1000
    np.savez(out_dir / "pi_verify_data.npz", windows=ds.windows, labels=ds.labels,
             expected_logits=np.array(expected, dtype=np.float32), names=np.array(ds.sample_ids))
    return len(ds), mac_ms


def realtime_budget(ms_per_chunk, chunks_per_window=8, stride_s=0.5):
    """윈도우가 0.5초마다 새로 생기므로, 윈도우 하나를 0.5초 안에 처리해야 밀리지 않는다."""
    per_window = ms_per_chunk * chunks_per_window
    return per_window, per_window / (stride_s * 1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="outputs/manual/onnx_clean/student.onnx")
    parser.add_argument("--processed-dir", default="outputs/manual/processed")
    parser.add_argument("--output-dir", default="outputs/manual/pi_bundle")
    args = parser.parse_args()

    files = []
    for code in ("K002", "KA04"):   # 5장에서 test로 뺀 베어링, 운전 조건별 1파일씩
        for cond in ("N09_M07_F10", "N15_M01_F10", "N15_M07_F04", "N15_M07_F10"):
            files += sorted((ROOT / args.processed_dir / code).glob(f"{cond}_*_windows.npy"))[:1]
    n, mac_ms = make_bundle(ROOT / args.onnx, files, ROOT / args.output_dir)
    out = ROOT / args.output_dir
    print(f"배포 묶음: {out.relative_to(ROOT)}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size / 1e6:6.1f} MB")
    print(f"\n검증용 윈도우 {n}개, 이 컴퓨터에서 윈도우당 {mac_ms:.1f} ms")
    for name, ms in (("이 컴퓨터", mac_ms / 8), ("Raspberry Pi 5 (실측)", 38.9)):
        per_window, load = realtime_budget(ms)
        print(f"  {name:22s} 청크 {ms:5.1f} ms → 윈도우 {per_window:6.1f} ms → 0.5초 예산의 {load:.0%} 사용 (100% 미만이면 실시간 가능)")


if __name__ == "__main__":
    main()
