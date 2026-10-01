"""8장 완성 코드: MOMENT에 LoRA를 붙이고, 기울기가 LoRA까지 실제로 전달되는지 검사한다.

이 프로젝트에서 실제로 겪은 버그도 재현한다:
    gradient checkpointing을 reentrant 방식으로 켜면 LoRA에 기울기가 전혀 가지 않는다.

실행 (프로젝트 루트에서):
    python docs/manual/code/ch08_lora.py                # Small로 빠르게
    python docs/manual/code/ch08_lora.py --size base    # 실제 Teacher와 같은 Base
"""

import argparse
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from ch07_moment import ROOT, load_moment

# 이 프로젝트의 LoRA 설정 (src/bearing_kd/models.py의 LORA_CONFIG)
LORA_CONFIG = {
    "r": 8,                      # 랭크: 보정 행렬의 "폭". 작을수록 가볍다
    "lora_alpha": 16,            # 보정의 세기 = alpha / r = 2배
    "lora_dropout": 0.05,        # 학습 중 LoRA 입력의 5%를 무작위로 끈다 (외우기 방지)
    "target_modules": ["q", "v"],  # attention의 query, value 행렬에만 붙인다
    "bias": "none",
    "task_type": "FEATURE_EXTRACTION",
}


def add_lora(model, config=LORA_CONFIG, reentrant=False):
    """LoRA를 주입하고 학습 대상(LoRA + head)만 requires_grad=True로 둔다."""
    from peft import LoraConfig, inject_adapter_in_model

    model = inject_adapter_in_model(LoraConfig(**config), model)
    # 메모리를 아끼는 gradient checkpointing. 반드시 use_reentrant=False (본문 8.5 참고)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("head.") or "lora_" in name
    return model


def expected_lora_params(model, r=8, targets=("q", "v")):
    """LoRA 파라미터 수를 손으로 계산: 행렬 W(out×in)마다 A(r×in) + B(out×r)."""
    total = 0
    for block in model.encoder.block:
        attn = block.layer[0].SelfAttention
        for t in targets:
            base = getattr(attn, t).base_layer if hasattr(getattr(attn, t), "base_layer") else getattr(attn, t)
            out_f, in_f = base.weight.shape
            total += r * in_f + out_f * r
    return total


def validate_gradients(model):
    """학습 대상인데 기울기가 없거나(None) NaN이면 즉시 멈춘다. 조용히 넘어가는 것이 가장 위험하다."""
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                raise RuntimeError(f"기울기 없음: {name}")
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(f"기울기에 NaN/무한대: {name}")


def one_backward(model):
    torch.manual_seed(0)
    model.train()
    x, y = torch.randn(4, 1, 512), torch.tensor([0, 1, 0, 1])
    loss = F.cross_entropy(model(x_enc=x, input_mask=torch.ones(4, 512)).logits, y)
    loss.backward()
    lora = [p for n, p in model.named_parameters() if "lora_" in n]
    return len(lora), sum(p.grad is not None for p in lora)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("small", "base"), default="small")
    args = parser.parse_args()
    model_dir = ROOT / f"models/MOMENT-1-{args.size}"

    model = add_lora(load_moment(model_dir))
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    head = sum(p.numel() for n, p in model.named_parameters() if n.startswith("head."))
    print(f"MOMENT-1-{args.size} + LoRA(r=8, q/v)")
    print(f"  LoRA 파라미터 {lora:,}개 (손 계산 {expected_lora_params(model):,}) + head {head:,}개 = 학습 대상 {lora + head:,}개")
    total = sum(p.numel() for p in model.parameters())
    print(f"  전체 {total:,}개 중 {100 * (lora + head) / total:.2f}%만 학습")

    print("\nreentrant 버그 재현 (역전파 1번 후 기울기를 받은 LoRA 파라미터 수)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for reentrant in (True, False):
            m = add_lora(load_moment(model_dir), reentrant=reentrant)
            n, got = one_backward(m)
            status = "정상" if got == n else "버그! LoRA가 학습되지 않음"
            print(f"  use_reentrant={reentrant!s:5s}: {got}/{n}  → {status}")
            try:
                validate_gradients(m)
                print("    validate_gradients 통과")
            except RuntimeError as e:
                print(f"    validate_gradients가 잡아냄: {e}")


if __name__ == "__main__":
    main()
