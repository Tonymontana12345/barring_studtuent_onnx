"""7장 완성 코드: MOMENT 사전학습 모델을 불러와 구조와 크기를 확인하고, 진동 청크를 넣어 본다.

실행 (프로젝트 루트에서):
    python docs/manual/code/ch07_moment.py                      # Small
    python docs/manual/code/ch07_moment.py --size base          # Base
"""

import argparse
import os
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 인터넷 없이 로컬 모델 폴더만 사용


def load_moment(model_dir, seed=42):
    """분류용 MOMENT를 만들고, 사전학습 가중치(몸통)는 모두 고정한다."""
    from momentfm import MOMENTPipeline

    # MOMENT는 "분류 head는 사전학습되지 않았다"는 경고를 낸다. 우리가 head를 학습하므로 정상이라 숨긴다.
    warnings.filterwarnings("ignore", message="Only reconstruction head is pre-trained")
    model = MOMENTPipeline.from_pretrained(
        str(model_dir), local_files_only=True,
        model_kwargs={
            "task_name": "classification",   # 분류용 머리(head)를 붙인다
            "n_channels": 1,                 # 진동 1채널
            "num_class": 2,                  # 정상 / 고장
            "freeze_encoder": True,          # 인코더(몸통) 고정
            "freeze_embedder": True,         # 패치 임베딩 고정
            "enable_gradient_checkpointing": False,  # 8장에서 설명
        },
    )
    torch.manual_seed(seed)                  # head 초기값을 매번 같게 (재현성)
    model.init()                             # 머리(head)를 새로 만든다 (무작위 초기값)
    return model


def count_params(model):
    """부분별 파라미터(학습 가능한 숫자) 개수."""
    parts = {}
    for name, p in model.named_parameters():
        key = name.split(".")[0]
        parts[key] = parts.get(key, 0) + p.numel()
    total = sum(parts.values())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    unused = model.encoder.embed_tokens.weight.numel()   # 글자용 임베딩: 시계열에서는 안 쓰임
    return parts, total, trainable, unused


def describe(model):
    parts, total, trainable, unused = count_params(model)
    cfg = model.encoder.config
    print(f"d_model(특징 길이) {model.config.d_model}, 층 {len(model.encoder.block)}개, "
          f"attention head {cfg.num_heads}개, 패치 길이 {model.patch_len}, 입력 길이 {model.seq_len}")
    for k, v in parts.items():
        print(f"  {k:16s} {v:>12,}")
    print(f"  {'합계':16s} {total:>12,}  (float32 기준 {total * 4 / 1e6:.0f} MB)")
    print(f"  그중 학습 가능  {trainable:>12,}  ← 머리(head)뿐")
    print(f"  그중 안 쓰는 글자 임베딩 {unused:,}개 (ONNX에서는 빠진다, 14장)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("small", "base"), default="small")
    args = parser.parse_args()
    model_dir = ROOT / f"models/MOMENT-1-{args.size}"
    model = load_moment(model_dir).eval()
    print(f"MOMENT-1-{args.size}  (파일 {os.path.getsize(model_dir / 'model.safetensors') / 1e6:.0f} MB)")
    describe(model)

    torch.manual_seed(0)
    x = torch.randn(3, 1, 512)                   # 청크 3개 [batch, 채널, 512]
    mask = torch.ones(3, 512)
    with torch.no_grad():
        out = model(x_enc=x, input_mask=mask)
    print(f"\n입력 {tuple(x.shape)} → logits {tuple(out.logits.shape)}")
    print("logits:", [[round(float(v), 3) for v in row] for row in out.logits], " ← 머리가 아직 학습 전이라 의미 없는 값")


if __name__ == "__main__":
    main()
