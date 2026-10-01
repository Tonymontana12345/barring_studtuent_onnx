"""11장 완성 코드: 지식 증류(KD). Teacher의 예측을 캐시로 저장하고, Student(MOMENT Small)의 head만 학습한다.

순서:
    1) cache_teacher : Teacher로 train/validation 윈도우를 예측해 logits를 .npz로 저장 (데이터 지문 포함)
    2) train_student : Small의 head만, "정답 + Teacher의 확신 정도"를 함께 보며 학습

실행 (프로젝트 루트에서, 9장 결과 사용):
    python docs/manual/code/ch11_distill.py
    # 전체 (실제 프로젝트 split, 프로젝트와 같은 설정)
    python docs/manual/code/ch11_distill.py --split data/splits/bearing_holdout_v3.json \
        --teacher outputs/manual/teacher_full/teacher_lora_best.pt --epochs 20 --learning-rate 1e-4 \
        --output-dir outputs/manual/student_full
"""

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ch06_dataset import WindowDataset
from ch07_moment import ROOT, load_moment
from ch09_train_teacher import evaluate, pick_device, seed_everything, window_logits
from ch10_checkpoint import load_full

TEMPERATURE = 2.0   # 확률 분포를 얼마나 "부드럽게" 펼칠지
HARD_WEIGHT = 0.5   # 정답 손실 비중 (나머지 0.5는 Teacher 따라하기)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def cache_teacher(teacher_ckpt, split, model_dir, out_dir, device):
    """Teacher 예측을 한 번만 계산해 저장한다. Student를 여러 번 학습해도 Teacher를 다시 돌릴 필요가 없다."""
    teacher = load_full(model_dir, teacher_ckpt).to(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation"):
        ds = WindowDataset([ROOT / f for f in split[name]])
        logits = torch.cat([window_logits(teacher, x.to(device)).cpu()
                            for x, _ in DataLoader(ds, batch_size=32)]).numpy()
        np.savez_compressed(out_dir / f"teacher_{name}.npz", logits=logits, labels=ds.labels,
                            sample_ids=np.array(ds.sample_ids), dataset_fingerprint=ds.fingerprint,
                            teacher_sha256=sha256(teacher_ckpt))
        acc = (logits.argmax(1) == ds.labels).mean()
        print(f"  teacher_{name}.npz: 윈도우 {len(ds)}개, Teacher 정확도 {acc:.3f}")


class DistillDataset(WindowDataset):
    """윈도우 + 정답 + 같은 윈도우에 대한 Teacher logits. 캐시와 데이터가 어긋나면 에러."""

    def __init__(self, files, cache_path):
        super().__init__(files)
        cache = np.load(cache_path)
        if str(cache["dataset_fingerprint"]) != self.fingerprint:
            raise ValueError("캐시를 만든 뒤 데이터가 바뀌었습니다. cache_teacher를 다시 실행하세요.")
        assert list(cache["sample_ids"]) == self.sample_ids and np.array_equal(cache["labels"], self.labels)
        self.teacher_logits = torch.from_numpy(cache["logits"]).float()

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return x, y, self.teacher_logits[i]


def kd_loss(student_logits, teacher_logits, labels, t=TEMPERATURE, hard_weight=HARD_WEIGHT):
    """0.5 × (정답 맞히기) + 0.5 × (Teacher의 확률 분포 따라하기)."""
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(F.log_softmax(student_logits / t, dim=1),   # Student 분포 (온도로 부드럽게)
                    F.softmax(teacher_logits / t, dim=1),        # Teacher 분포 (온도로 부드럽게)
                    reduction="batchmean") * t ** 2              # t²: 온도 때문에 작아진 기울기 크기 보정
    return hard_weight * hard + (1 - hard_weight) * soft


def train_student(split, cache_dir, model_dir, out_dir, epochs, lr, batch_size, seed, device):
    seed_everything(seed)
    train_ds = DistillDataset([ROOT / f for f in split["train"]], cache_dir / "teacher_train.npz")
    valid_ds = WindowDataset([ROOT / f for f in split["validation"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        student = load_moment(model_dir, seed=seed).to(device)   # 몸통 고정, head만 학습 (7장)
    params = [p for p in student.parameters() if p.requires_grad]
    print(f"  Student 학습 대상 {sum(p.numel() for p in params):,}개 (head만)")
    optimizer = torch.optim.AdamW(params, lr=lr)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    (out_dir / "epochs").mkdir(parents=True, exist_ok=True)
    best = (-1.0, float("-inf"))
    for epoch in range(1, epochs + 1):
        student.train()
        total = 0.0
        for x, y, t_logits in loader:
            x, y, t_logits = x.to(device), y.to(device), t_logits.to(device)
            loss = kd_loss(window_logits(student, x), t_logits, y)   # Student는 윈도우 단위(8청크 평균)로 학습
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        result = evaluate(student, valid_ds, device)
        head = {k: v.detach().cpu().clone() for k, v in student.state_dict().items() if k.startswith("head.")}
        torch.save({"epoch": epoch, "trainable_state_dict": head, "validation": result},
                   out_dir / "epochs" / f"epoch_{epoch:02d}.pt")
        score = (result["accuracy"], -result["loss"])
        mark = ""
        if score > best:
            best, mark = score, "  ← best 저장"
            torch.save({"epoch": epoch, "model_state_dict": student.state_dict(), "validation_accuracy": result["accuracy"],
                        "teacher_sha256": str(np.load(cache_dir / "teacher_train.npz")["teacher_sha256"])},
                       out_dir / "student_kd_best.pt")
        if epoch <= 3 or epoch % 5 == 0 or epoch == epochs or mark:
            print(f"  epoch {epoch:2d}  KD loss {total / len(train_ds):.4f}  validation {result['correct']}/{result['total']} "
                  f"loss {result['loss']:.4f} {result['confusion_matrix']}{mark}")
    return student


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="outputs/manual/split_holdout.json")
    parser.add_argument("--teacher", default="outputs/manual/teacher/teacher_lora_best.pt")
    parser.add_argument("--teacher-model-dir", default="models/MOMENT-1-base")
    parser.add_argument("--student-model-dir", default="models/MOMENT-1-small")
    parser.add_argument("--output-dir", default="outputs/manual/student")
    parser.add_argument("--epochs", type=int, default=20)
    # head 1,026개만 학습하고 맛보기 데이터는 epoch당 6 step뿐이라 크게 1e-2. 전체 데이터는 프로젝트처럼 1e-4
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    split = json.loads((ROOT / args.split).read_text())
    out_dir = ROOT / args.output_dir
    cache_dir = out_dir / "cache"
    print("[1] Teacher 예측 캐시")
    cache_teacher(ROOT / args.teacher, split, ROOT / args.teacher_model_dir, cache_dir, device)
    print("[2] Student(head) 증류 학습")
    train_student(split, cache_dir, ROOT / args.student_model_dir, out_dir, args.epochs,
                  args.learning_rate, args.batch_size, args.seed, device)


if __name__ == "__main__":
    main()
