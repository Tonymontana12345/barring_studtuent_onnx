"""Export a fixed-batch chunk model and verify numerical parity."""

from pathlib import Path
import copy
import json
import numpy as np
import torch
import onnx

from .models import load_model
from .inference import create_session, chunk_window
from .data import sha256_file

class ExportableRevIN(
    torch.nn.Module
):
    """
    MOMENT의 RevIN을 ONNX 호환 연산으로 구현한다.
    nanmean, nanvar를 사용하지 않는다.
    """

    def __init__(
        self,
        original_revin,
    ):
        super().__init__()

        self.eps = original_revin.eps
        self.affine = original_revin.affine

        if self.affine:
            self.affine_weight = (
                original_revin.affine_weight
            )
            self.affine_bias = (
                original_revin.affine_bias
            )

    def forward(
        self,
        x,
        mode="norm",
        mask=None,
    ):
        if mode != "norm":
            raise RuntimeError(
                "ExportableRevIN은 "
                "norm 모드만 지원합니다."
            )

        if mask is None:
            mask = torch.ones(
                x.shape[0],
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )

        mask = mask.unsqueeze(1)
        mask = mask.expand_as(x)

        valid = (mask > 0) & torch.isfinite(x)

        safe_x = torch.where(
            valid,
            x,
            torch.zeros_like(x),
        )

        count = valid.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1)

        count = count.to(x.dtype)

        mean = (
            safe_x.sum(
                dim=-1,
                keepdim=True,
            )
            / count
        )

        centered = torch.where(
            valid,
            x - mean,
            torch.zeros_like(x),
        )

        variance = (
            centered.square().sum(
                dim=-1,
                keepdim=True,
            )
            / count
        )

        stdev = torch.sqrt(
            variance
        ) + self.eps

        output = (
            x - mean
        ) / stdev

        if self.affine:
            output = (
                output
                * self.affine_weight
            )

            output = (
                output
                + self.affine_bias
            )

        return output


class ExportableClassificationHead(
    torch.nn.Module
):
    """
    MOMENT ClassificationHead의 평가 동작을 ONNX 연산으로 구현한다.
    """

    def __init__(
        self,
        original_head,
    ):
        super().__init__()

        # 학습된 Classification Head의
        # Linear 가중치를 그대로 사용한다.
        self.linear = original_head.linear

    def forward(
        self,
        x,
        input_mask=None,
    ):
        valid = ~torch.isnan(x)

        safe_x = torch.where(
            valid,
            x,
            torch.zeros_like(x),
        )

        valid_count = valid.sum(
            dim=1,
            keepdim=False,
        ).clamp_min(1)

        valid_count = valid_count.to(
            x.dtype
        )

        pooled = (
            safe_x.sum(dim=1)
            / valid_count
        )

        return self.linear(pooled)


class StudentChunkWrapper(
    torch.nn.Module
):
    """
    MOMENT의 TimeseriesOutputs에서
    logits만 반환하도록 감싼다.
    """

    def __init__(
        self,
        student,
    ):
        super().__init__()

        self.student = student

    def forward(
        self,
        x_enc,
        input_mask,
    ):
        outputs = self.student(
            x_enc=x_enc,
            input_mask=input_mask,
        )

        return outputs.logits


def export_student(model_dir, checkpoint, output_path, sample_window=None):
    output_path = Path(output_path)
    if output_path.exists() or Path(str(output_path) + ".data").exists():
        raise FileExistsError(f"Use a new ONNX output path: {output_path}")
    original = load_model(model_dir, checkpoint)
    student = copy.deepcopy(original)
    student.normalizer = ExportableRevIN(student.normalizer)
    student.head = ExportableClassificationHead(student.head)
    wrapper = StudentChunkWrapper(student).eval()
    generator = torch.Generator().manual_seed(42)
    dummy_x = torch.randn(1, 1, 512, generator=generator)
    dummy_mask = torch.ones(1, 512)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(wrapper, (dummy_x, dummy_mask), str(output_path),
                          input_names=["x_enc", "input_mask"], output_names=["logits"],
                          opset_version=18, dynamo=True, external_data=True)
    onnx.checker.check_model(str(output_path))
    session = create_session(output_path)
    cases = [(dummy_x, dummy_mask), (torch.zeros_like(dummy_x), dummy_mask),
             (torch.ones_like(dummy_x), dummy_mask)]
    padded_x = dummy_x.clone()
    padded_x[:, :, 416:] = 0
    padded_mask = dummy_mask.clone()
    padded_mask[:, 416:] = 0
    cases.append((padded_x, padded_mask))
    if sample_window is not None:
        chunks, masks = chunk_window(sample_window)
        cases.extend((torch.from_numpy(chunks[i:i+1]), torch.from_numpy(masks[i:i+1])) for i in range(8))
    maximum = 0.
    with torch.no_grad():
        for x, mask in cases:
            expected = original(x_enc=x, input_mask=mask).logits.numpy()
            replaced = wrapper(x, mask).numpy()
            actual = session.run(["logits"], {"x_enc": x.numpy(), "input_mask": mask.numpy()})[0]
            np.testing.assert_allclose(replaced, expected, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    result = {"checkpoint_sha256": sha256_file(checkpoint), "cases": len(cases),
              "max_absolute_error": maximum, "onnx": str(output_path)}
    output_path.with_suffix(".verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

