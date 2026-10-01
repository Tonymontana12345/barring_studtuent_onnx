"""12장 완성 코드: test 세트로 모델을 평가하고, 베어링별·운전 조건별로 나눠 본다.

윈도우 판정 = 청크 8개의 logits 평균 → argmax (9장 window_logits)

실행 (프로젝트 루트에서):
    # 맛보기: 9·11장 결과를 5장 test(K002, KA04)로
    python docs/manual/code/ch12_evaluate.py
    # 실제 프로젝트 v3 Student를 실제 test(K004, KB24, KB27)로
    python docs/manual/code/ch12_evaluate.py --split data/splits/bearing_holdout_v3.json \
        --student outputs/runs/v3/student/student_kd_best.pt --teacher none
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ch06_dataset import WindowDataset
from ch07_moment import ROOT
from ch09_train_teacher import pick_device, window_logits
from ch10_checkpoint import load_full


@torch.no_grad()
def predict(model, dataset, device, batch_size=32):
    model.to(device).eval()
    return torch.cat([window_logits(model, x.to(device)).cpu()
                      for x, _ in DataLoader(dataset, batch_size=batch_size)]).numpy()


def metrics(logits, labels):
    pred = logits.argmax(1)
    confusion = np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (labels, pred), 1)
    tn, fp, fn, tp = confusion.ravel()
    return {"accuracy": float((pred == labels).mean()), "correct": int((pred == labels).sum()), "total": len(labels),
            "confusion_matrix": confusion.tolist(),
            "false_alarm_rate": float(fp / max(tn + fp, 1)),   # 정상인데 고장이라고 한 비율
            "miss_rate": float(fn / max(fn + tp, 1))}          # 고장인데 정상이라고 한 비율 (가장 위험)


def breakdown(logits, dataset):
    """베어링별, 베어링×운전 조건별 정답 수."""
    pred = logits.argmax(1)
    by_bearing, by_condition = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for sid, p, y in zip(dataset.sample_ids, pred, dataset.labels):
        bearing, name = sid.split("/")[0], sid.split("/")[1]
        for table, key in ((by_bearing, bearing), (by_condition, (bearing, name[:11]))):
            table[key][0] += int(p == y)
            table[key][1] += 1
    return by_bearing, by_condition


def report(name, logits, dataset):
    m = metrics(logits, dataset.labels)
    print(f"\n[{name}] 정확도 {m['correct']}/{m['total']} = {m['accuracy']:.1%}  혼동행렬 {m['confusion_matrix']}")
    print(f"  오경보율 {m['false_alarm_rate']:.1%}  놓침률 {m['miss_rate']:.1%}")
    by_bearing, by_condition = breakdown(logits, dataset)
    for bearing, (ok, n) in sorted(by_bearing.items()):
        conds = "  ".join(f"{c}:{by_condition[(bearing, c)][0]}/{by_condition[(bearing, c)][1]}"
                          for b, c in sorted(by_condition) if b == bearing)
        print(f"  {bearing}: {ok}/{n} ({ok / n:.0%})   {conds}")
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="outputs/manual/split_holdout.json")
    parser.add_argument("--teacher", default="outputs/manual/teacher/teacher_lora_best.pt", help="none이면 생략")
    parser.add_argument("--student", default="outputs/manual/student/student_kd_best.pt", help="none이면 생략")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    split = json.loads((ROOT / args.split).read_text())
    test = WindowDataset([ROOT / f for f in split["test"]])
    print(f"test: 윈도우 {len(test)}개, 베어링 {sorted({s.split('/')[0] for s in test.sample_ids})}")
    results = {}
    if args.teacher != "none":
        results["teacher"] = report("Teacher", predict(load_full(ROOT / "models/MOMENT-1-base", ROOT / args.teacher), test, device), test)
    if args.student != "none":
        results["student"] = report("Student", predict(load_full(ROOT / "models/MOMENT-1-small", ROOT / args.student, teacher=False), test, device), test)
    return results


if __name__ == "__main__":
    main()
