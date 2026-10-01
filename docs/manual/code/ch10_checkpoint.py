"""10장 완성 코드: 체크포인트(.pt) 안을 들여다보고, 다시 모델로 불러온다.

- 전체 체크포인트(teacher_lora_best.pt)를 strict 모드로 불러오기
- epoch별로 저장한 "학습된 부분만" 체크포인트를 사전학습 모델 위에 덮어 불러오기
- LoRA B 행렬이 0인지 검사해 LoRA가 실제로 학습됐는지 확인하기

실행 (프로젝트 루트에서, 9장 결과 사용):
    python docs/manual/code/ch10_checkpoint.py
    python docs/manual/code/ch10_checkpoint.py --checkpoint outputs/checkpoints/teacher_lora_best.pt   # 옛날 모델
"""

import argparse
import warnings
from pathlib import Path

import torch

from ch07_moment import ROOT, load_moment
from ch08_lora import LORA_CONFIG, add_lora


def load_checkpoint(path):
    # weights_only=True: 텐서와 기본 자료형만 읽는다. 남이 만든 .pt 안의 악성 코드 실행을 막는다.
    return torch.load(path, map_location="cpu", weights_only=True)


def describe(path):
    ckpt = load_checkpoint(path)
    print(f"{Path(path).relative_to(ROOT)}  ({Path(path).stat().st_size / 1e6:.1f} MB)")
    for key, value in ckpt.items():
        if isinstance(value, dict):
            tensors = [v for v in value.values() if torch.is_tensor(v)]
            n = sum(t.numel() for t in tensors)
            print(f"  {key:22s} dict, 항목 {len(value)}개" + (f", 숫자 {n:,}개 ({n * 4 / 1e6:.1f} MB)" if tensors else ""))
        else:
            print(f"  {key:22s} {value}")
    return ckpt


def lora_b_report(state):
    """LoRA B는 0으로 시작한다. 학습이 됐다면 0이 아닌 값이 있어야 한다."""
    b = {k: v for k, v in state.items() if "lora_B" in k}
    nonzero = sum(int(v.abs().sum() > 0) for v in b.values())
    return nonzero, len(b)


def load_full(model_dir, path, teacher=True):
    """전체 state_dict를 strict=True로 불러온다. 이름 하나라도 안 맞으면 에러 → 조용한 실수를 막는다."""
    ckpt = load_checkpoint(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = load_moment(model_dir)
        if teacher:
            model = add_lora(model, ckpt.get("lora_config", LORA_CONFIG))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.eval()


def load_from_epoch(model_dir, path):
    """epoch별 파일(LoRA + head만)을 사전학습 Base 위에 덮어쓴다. 나머지는 사전학습 값 그대로."""
    part = load_checkpoint(path)["trainable_state_dict"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = add_lora(load_moment(model_dir))
    result = model.load_state_dict(part, strict=False)
    # strict=False일 때는 결과를 직접 확인해야 한다: 모르는 이름(unexpected)이 있으면 안 된다
    assert not result.unexpected_keys, result.unexpected_keys
    missing_trainable = [k for k in result.missing_keys if "lora_" in k or k.startswith("head.")]
    assert not missing_trainable, missing_trainable
    return model.eval(), len(part), len(result.missing_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/manual/teacher/teacher_lora_best.pt")
    parser.add_argument("--model-dir", default="models/MOMENT-1-base")
    args = parser.parse_args()

    ckpt = describe(ROOT / args.checkpoint)
    nonzero, total = lora_b_report(ckpt["model_state_dict"])
    print(f"\nLoRA B 행렬 중 0이 아닌 것: {nonzero}/{total}  → {'학습됨' if nonzero == total else 'LoRA가 학습되지 않음!'}")

    epochs_dir = (ROOT / args.checkpoint).parent / "epochs"
    if epochs_dir.exists():
        model_dir = ROOT / args.model_dir
        full = load_full(model_dir, ROOT / args.checkpoint)
        best_epoch = ckpt["epoch"]
        part_model, n_part, n_missing = load_from_epoch(model_dir, epochs_dir / f"epoch_{best_epoch:02d}.pt")
        torch.manual_seed(0)
        x = torch.randn(4, 1, 512)
        with torch.no_grad():
            same = torch.allclose(full(x_enc=x).logits, part_model(x_enc=x).logits, atol=1e-6)
        print(f"\nepoch_{best_epoch:02d}.pt: 학습된 텐서 {n_part}개만 덮어씀 (나머지 {n_missing}개는 사전학습 값)")
        print(f"전체 체크포인트와 예측이 같은가? {same}")


if __name__ == "__main__":
    main()
