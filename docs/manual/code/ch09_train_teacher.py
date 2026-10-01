"""9장 완성 코드: Teacher(MOMENT Base + LoRA + head)를 학습한다.

src/bearing_kd/training.py의 train(teacher=True)을 읽기 쉽게 줄인 버전이다.
프로젝트 코드와 다른 점 (프로젝트에서 겪은 문제를 반영한 개선):
    - 최고 모델은 (validation 정확도, -validation loss) 순으로 비교한다.
      정확도가 100%에 붙어 같아져도 loss가 더 낮은(더 확신하는) epoch를 고른다.
    - epoch마다 학습된 부분(LoRA + head)만 작게 따로 저장한다 → 원하는 epoch로 되돌릴 수 있다.

실행 (프로젝트 루트에서):
    # 맛보기: 4·5장 결과(작은 split), 3 epoch, learning rate 1e-3 (약 1분)
    python docs/manual/code/ch09_train_teacher.py
    # 전체: 실제 프로젝트 split, 프로젝트와 같은 설정 (Mac MPS 기준 약 3.5시간)
    PYTORCH_ENABLE_MPS_FALLBACK=1 python docs/manual/code/ch09_train_teacher.py \
        --split data/splits/bearing_holdout_v3.json --epochs 20 --learning-rate 1e-4 --output-dir outputs/manual/teacher_full
"""

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ch06_dataset import WindowDataset, make_moment_chunks
from ch07_moment import ROOT, load_moment
from ch08_lora import LORA_CONFIG, add_lora, validate_gradients


def seed_everything(seed):
    """무작위성이 들어가는 모든 곳의 시작점을 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name="auto"):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def window_logits(model, batch_x):
    """윈도우 [B,1,4000] → 청크 8개씩 예측 → 8개 logits 평균 → [B,2]."""
    chunks, mask, _ = make_moment_chunks(batch_x)
    logits = model(x_enc=chunks, input_mask=mask).logits
    return logits.reshape(batch_x.shape[0], -1, 2).mean(dim=1)


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=32):
    model.eval()
    all_logits = []
    for x, _ in DataLoader(dataset, batch_size=batch_size):
        all_logits.append(window_logits(model, x.to(device)).cpu())
    logits = torch.cat(all_logits)
    labels = dataset.labels
    loss = F.cross_entropy(logits, torch.from_numpy(labels)).item()   # 윈도우 단위 validation loss
    pred = logits.argmax(1).numpy()
    confusion = np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (labels, pred), 1)
    return {"accuracy": float((pred == labels).mean()), "correct": int((pred == labels).sum()),
            "total": len(labels), "loss": loss, "confusion_matrix": confusion.tolist()}


def trainable_state(model):
    """학습된 부분(LoRA + head)만 뽑는다. 전체 457 MB 대신 약 1 MB."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k or k.startswith("head.")}


def train_teacher(split_path, model_dir, output_dir, epochs=3, batch_size=16, lr=1e-3, seed=42, device="auto"):
    device = pick_device(device)
    seed_everything(seed)
    split = json.loads(Path(split_path).read_text())
    train_ds = WindowDataset([ROOT / f for f in split["train"]])
    valid_ds = WindowDataset([ROOT / f for f in split["validation"]])
    print(f"장치 {device} | train {len(train_ds)} 윈도우, validation {len(valid_ds)} 윈도우")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = add_lora(load_moment(model_dir, seed=seed)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    print(f"학습 대상 파라미터 {sum(p.numel() for p in params):,}개")

    output_dir = Path(output_dir)
    (output_dir / "epochs").mkdir(parents=True, exist_ok=True)
    config = {"model_dir": str(Path(model_dir).relative_to(ROOT)), "split": str(Path(split_path).relative_to(ROOT)),
              "epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "seed": seed, "device": device,
              "lora_config": LORA_CONFIG, "train_fingerprint": train_ds.fingerprint}
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best, history = (-1.0, float("-inf")), []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        total_loss, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            chunks, mask, chunk_y = make_moment_chunks(x, y)   # 윈도우 → 청크 8개, 라벨도 8번 복사
            logits = model(x_enc=chunks, input_mask=mask).logits
            loss = F.cross_entropy(logits, chunk_y)            # Teacher는 청크마다 정답을 맞히도록 학습
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            validate_gradients(model)                          # LoRA 기울기가 없으면 여기서 멈춘다
            optimizer.step()
            total_loss += loss.item() * len(y)
            count += len(y)
        result = evaluate(model, valid_ds, device)
        record = {"epoch": epoch, "train_loss": total_loss / count, "validation": result,
                  "seconds": round(time.time() - start, 1)}
        history.append(record)
        torch.save({"epoch": epoch, "trainable_state_dict": trainable_state(model), "validation": result},
                   output_dir / "epochs" / f"epoch_{epoch:02d}.pt")
        score = (result["accuracy"], -result["loss"])   # 정확도 우선, 같으면 loss가 낮은 쪽
        improved = score > best
        if improved:
            best = score
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "validation_accuracy": result["accuracy"],
                        "lora_config": LORA_CONFIG, "training_config": config},
                       output_dir / "teacher_lora_best.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch {epoch:2d}  train loss {record['train_loss']:.4f}  validation {result['correct']}/{result['total']} "
              f"loss {result['loss']:.4f} {result['confusion_matrix']}  {record['seconds']}s{'  ← best 저장' if improved else ''}")
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="outputs/manual/split_holdout.json")
    parser.add_argument("--model-dir", default="models/MOMENT-1-base")
    parser.add_argument("--output-dir", default="outputs/manual/teacher")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    # 맛보기는 데이터가 작아(epoch당 6 step) 1e-3. 전체 데이터(epoch당 179 step)는 프로젝트처럼 1e-4
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / mps")
    args = parser.parse_args()
    train_teacher(ROOT / args.split, ROOT / args.model_dir, ROOT / args.output_dir, args.epochs,
                  args.batch_size, args.learning_rate, args.seed, args.device)


if __name__ == "__main__":
    main()
