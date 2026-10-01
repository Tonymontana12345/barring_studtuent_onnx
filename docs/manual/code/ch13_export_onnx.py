"""13장 완성 코드: Student 체크포인트를 ONNX로 변환하고, PyTorch와 결과가 같은지 검증한다.

src/bearing_kd/export.py를 읽기 쉽게 줄인 버전이다.
    1) 체크포인트 불러오기 (MOMENT Small + 학습된 head)
    2) RevIN, head를 ONNX 친화적인 구현으로 교체 (결과는 같음)
    3) logits만 돌려주는 wrapper로 감싸기
    4) torch.onnx.export (dynamo 방식, opset 18, 가중치는 .onnx.data로 분리)
    5) onnx.checker로 형식 검사 + onnxruntime 결과를 PyTorch와 비교

실행 (프로젝트 루트에서, 11장 결과 사용):
    python docs/manual/code/ch13_export_onnx.py --overwrite
    python docs/manual/code/ch13_export_onnx.py --checkpoint outputs/runs/v3/student/student_kd_best.pt \
        --output outputs/manual/onnx_v3/student.onnx --overwrite
"""

import argparse
import copy
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch

from ch07_moment import ROOT
from ch10_checkpoint import load_full

CHUNK = 512
logging.getLogger("torch.onnx").setLevel(logging.ERROR)   # "torchvision 없음" 같은 무관한 경고 숨김


class ExportableRevIN(torch.nn.Module):
    """MOMENT RevIN과 같은 계산을 nanmean 없이 한다.

    원본은 mask=0인 칸을 NaN으로 바꾼 뒤 nanmean/nanstd를 쓴다. ONNX에는 nanmean 연산이 없어서
    옛 방식(TorchScript) exporter는 변환에 실패한다. 여기서는 "유효한 칸만 더하고 개수로 나누기"로 바꾼다.
    """

    def __init__(self, original):
        super().__init__()
        self.eps = original.eps

    def forward(self, x, mode="norm", mask=None):
        mask = torch.ones(x.shape[0], x.shape[-1], dtype=x.dtype) if mask is None else mask
        valid = (mask.unsqueeze(1).expand_as(x) > 0) & torch.isfinite(x)       # 유효한 칸
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1).to(x.dtype)
        mean = torch.where(valid, x, torch.zeros_like(x)).sum(dim=-1, keepdim=True) / count
        var = torch.where(valid, x - mean, torch.zeros_like(x)).square().sum(dim=-1, keepdim=True) / count
        return (x - mean) / (torch.sqrt(var) + self.eps)


class ExportableHead(torch.nn.Module):
    """분류 head: 패치 특징을 평균 → 선형층. 평가 모드에서는 dropout이 꺼지므로 빼고, NaN이 섞여도 안전한 평균을 쓴다."""

    def __init__(self, original):
        super().__init__()
        self.linear = original.linear          # 학습된 가중치를 그대로 사용

    def forward(self, x, input_mask=None):
        valid = ~torch.isnan(x)
        pooled = torch.where(valid, x, torch.zeros_like(x)).sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(x.dtype)
        return self.linear(pooled)


class ChunkWrapper(torch.nn.Module):
    """MOMENT는 결과를 여러 필드가 든 객체로 돌려준다. ONNX 출력은 텐서여야 하므로 logits만 꺼낸다."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_enc, input_mask):
        return self.model(x_enc=x_enc, input_mask=input_mask).logits


def build_exportable(checkpoint):
    original = load_full(ROOT / "models/MOMENT-1-small", checkpoint, teacher=False)
    model = copy.deepcopy(original)
    model.normalizer = ExportableRevIN(model.normalizer)
    model.head = ExportableHead(model.head)
    return original, ChunkWrapper(model).eval()


def parity_cases(sample_window=None):
    """변환이 맞는지 볼 입력들: 무작위, 0, 상수, 마지막 96칸 패딩, 실제 윈도우의 청크 8개."""
    g = torch.Generator().manual_seed(42)
    x = torch.randn(1, 1, CHUNK, generator=g)
    ones = torch.ones(1, CHUNK)
    padded_x, padded_mask = x.clone(), ones.clone()
    padded_x[..., 416:], padded_mask[:, 416:] = 0, 0
    cases = [("random", x, ones), ("zeros", torch.zeros_like(x), ones), ("constant", torch.ones_like(x), ones),
             ("padded", padded_x, padded_mask)]
    if sample_window is not None:
        w = np.pad(np.asarray(sample_window, dtype=np.float32), (0, 96)).reshape(8, 1, CHUNK)
        m = (np.arange(8 * CHUNK) < 4000).astype(np.float32).reshape(8, CHUNK)
        cases += [(f"real chunk {i + 1}", torch.from_numpy(w[i:i + 1]), torch.from_numpy(m[i:i + 1])) for i in range(8)]
    return cases


def export(checkpoint, output, sample_window=None):
    import onnx
    import onnxruntime as ort

    original, wrapper = build_exportable(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    dummy = (torch.randn(1, 1, CHUNK), torch.ones(1, CHUNK))   # 입력 모양의 예시 (값은 상관없음)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(wrapper, dummy, str(output), input_names=["x_enc", "input_mask"], output_names=["logits"],
                          opset_version=18, dynamo=True, external_data=True, verbose=False)
    onnx.checker.check_model(str(output))                       # ONNX 형식 규칙을 지켰는지 검사
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])

    worst = 0.0
    print(f"\n{'입력':14s} {'PyTorch 원본':>22s} {'ONNX Runtime':>22s}  차이")
    with torch.no_grad():
        for name, x, mask in parity_cases(sample_window):
            expected = original(x_enc=x, input_mask=mask).logits.numpy()
            actual = session.run(["logits"], {"x_enc": x.numpy(), "input_mask": mask.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)   # 허용 오차를 넘으면 에러
            diff = float(np.abs(actual - expected).max())
            worst = max(worst, diff)
            fmt = lambda a: "[" + ", ".join(f"{v:+.4f}" for v in a[0]) + "]"
            print(f"{name:14s} {fmt(expected):>22s} {fmt(actual):>22s}  {diff:.1e}")
    result = {"checkpoint": str(Path(checkpoint).relative_to(ROOT)), "onnx": str(output.relative_to(ROOT)),
              "cases": len(parity_cases(sample_window)), "max_absolute_error": worst}
    output.with_suffix(".verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/manual/student/student_kd_best.pt")
    parser.add_argument("--output", default="outputs/manual/onnx/student.onnx")
    parser.add_argument("--sample", default="outputs/manual/processed/KA04/N09_M07_F10_KA04_1_windows.npy",
                        help="실제 윈도우로도 비교 (없으면 생략)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = ROOT / args.output
    data = Path(str(output) + ".data")
    if output.exists() or data.exists():
        if not args.overwrite:
            raise SystemExit(f"{output.relative_to(ROOT)} 가 이미 있습니다. --overwrite 를 쓰거나 다른 경로를 주세요.")
        output.unlink(missing_ok=True)
        data.unlink(missing_ok=True)
    sample = ROOT / args.sample
    window = np.load(sample)[0] if sample.exists() else None
    result = export(ROOT / args.checkpoint, output, window)
    size = output.stat().st_size / 1e6, data.stat().st_size / 1e6
    print(f"\n저장: {output.relative_to(ROOT)} ({size[0]:.1f} MB) + {data.name} ({size[1]:.1f} MB)")
    print(f"검증 {result['cases']}개 입력 통과, 최대 차이 {result['max_absolute_error']:.1e}")


if __name__ == "__main__":
    main()
