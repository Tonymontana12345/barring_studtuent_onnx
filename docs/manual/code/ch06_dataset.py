"""6장 완성 코드: split JSON → PyTorch Dataset → DataLoader → MOMENT 입력(512 × 8 청크 + mask).

src/bearing_kd/data.py의 BearingWindowDataset, make_moment_chunks를 읽기 쉽게 줄인 버전이다.

실행 (프로젝트 루트에서, 4·5장 결과 사용):
    python docs/manual/code/ch06_dataset.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
WINDOW_SIZE = 4_000
CHUNK_SIZE = 512       # MOMENT가 한 번에 받는 길이 (고정)


class WindowDataset(Dataset):
    """윈도우 파일 목록을 읽어 (신호, 라벨) 쌍을 하나씩 꺼내 주는 상자."""

    def __init__(self, window_files):
        windows, labels, sample_ids = [], [], []
        digest = hashlib.sha256()
        for path in window_files:
            path = Path(path)
            w = np.load(path)                                          # [파일당 윈도우 수, 4000]
            y = np.load(path.with_name(path.name.replace("_windows", "_labels")))
            assert w.shape[1] == WINDOW_SIZE and len(w) == len(y)
            windows.append(w.astype(np.float32))
            labels.append(y.astype(np.int64))
            sample_ids += [f"{path.parent.name}/{path.name}:{i}" for i in range(len(w))]
            digest.update(path.name.encode())
            digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        self.windows = np.concatenate(windows)   # [전체 윈도우 수, 4000]
        self.labels = np.concatenate(labels)     # [전체 윈도우 수]
        self.sample_ids = sample_ids             # 각 윈도우가 어느 파일 몇 번째인지
        self.fingerprint = digest.hexdigest()    # 데이터 내용이 바뀌면 이 값도 바뀐다

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        x = torch.from_numpy(self.windows[index]).unsqueeze(0)  # [1, 4000]: 채널 1개
        y = torch.tensor(self.labels[index])
        return x, y


def make_moment_chunks(batch_x, batch_y=None, chunk=CHUNK_SIZE):
    """[B, 1, 4000] → [B*8, 1, 512] 청크 + [B*8, 512] mask.

    4000은 512로 나누어떨어지지 않으므로 4096(=512×8)까지 0을 채우고,
    채운 96칸은 mask=0으로 표시해 모델이 무시하게 한다.
    """
    b, c, length = batch_x.shape
    count = -(-length // chunk)                                  # 올림 나눗셈: 8
    padded = F.pad(batch_x, (0, count * chunk - length))          # [B, 1, 4096]
    chunks = padded.reshape(b, c, count, chunk).permute(0, 2, 1, 3).reshape(b * count, c, chunk)
    # mask는 입력과 같은 장치(CPU/GPU)에 만들어야 한다. 다르면 "Expected all tensors to be on the same device" 에러
    mask = (torch.arange(count * chunk, device=batch_x.device) < length).to(batch_x.dtype).reshape(count, chunk).repeat(b, 1)
    labels = None if batch_y is None else batch_y.repeat_interleave(count)
    return chunks, mask, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="outputs/manual/split_holdout.json")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    split = json.loads((ROOT / args.split).read_text())
    datasets = {name: WindowDataset([ROOT / f for f in files]) for name, files in split.items() if files}
    for name, ds in datasets.items():
        counts = np.bincount(ds.labels, minlength=2)
        print(f"{name:10s} 윈도우 {len(ds):4d}개 (정상 {counts[0]}, 고장 {counts[1]})  지문 {ds.fingerprint[:12]}…")

    x, y = datasets["train"][0]
    print(f"\n샘플 1개: x {tuple(x.shape)} {x.dtype}, y = {y.item()}  ← {datasets['train'].sample_ids[0]}")

    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    batch_x, batch_y = next(iter(loader))
    print(f"batch 1개: x {tuple(batch_x.shape)}, y {tuple(batch_y.shape)} → 라벨 {batch_y.tolist()}")
    print(f"한 epoch = batch {len(loader)}번")

    chunks, mask, chunk_y = make_moment_chunks(batch_x, batch_y)
    print(f"\nMOMENT 입력: chunks {tuple(chunks.shape)}, mask {tuple(mask.shape)}, 청크 라벨 {tuple(chunk_y.shape)}")
    print(f"윈도우 1개의 mask 합계 = {mask[:8].sum().item():.0f} (실제 샘플 4000개, 패딩 96개는 0)")
    restored = chunks[:8, 0].reshape(-1)[:WINDOW_SIZE]
    print(f"청크 8개를 이어 붙이면 원래 윈도우와 같은가? {torch.equal(restored, batch_x[0, 0])}")


if __name__ == "__main__":
    main()
