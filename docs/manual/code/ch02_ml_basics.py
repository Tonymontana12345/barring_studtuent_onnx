"""2장 완성 코드: 가장 작은 분류 모델을 처음부터 학습해 보면서 ML 기본 개념을 확인한다.

- 가짜 데이터(점 2개 묶음)를 만들고
- train / validation / test로 나누고
- 선형 분류기를 epoch, batch 단위로 학습하고
- logits → softmax → 정확도, 혼동행렬로 평가한다.
- 마지막으로 "정답을 무작위로 섞은" 데이터로 오버피팅을 눈으로 확인한다.

실행 (프로젝트 루트에서):
    python docs/manual/code/ch02_ml_basics.py
"""

import numpy as np
import torch
import torch.nn.functional as F

SEED = 42


def make_data(n=600, seed=SEED):
    """정상(0)은 (0,0) 근처, 고장(1)은 (2,2) 근처에 흩어진 2차원 점."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 0.8, size=(n // 2, 2))
    x1 = rng.normal(2.0, 0.8, size=(n // 2, 2))
    x = np.concatenate([x0, x1]).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    order = rng.permutation(n)
    return torch.from_numpy(x[order]), torch.from_numpy(y[order])


def split(x, y, train=0.7, valid=0.15):
    a, b = int(len(x) * train), int(len(x) * (train + valid))
    return (x[:a], y[:a]), (x[a:b], y[a:b]), (x[b:], y[b:])


def confusion(pred, true):
    """행 = 실제, 열 = 예측. [[정상→정상, 정상→고장], [고장→정상, 고장→고장]]"""
    m = np.zeros((2, 2), dtype=int)
    for p, t in zip(pred.tolist(), true.tolist()):
        m[t, p] += 1
    return m


def train(model, train_set, valid_set, epochs=20, batch_size=32, lr=0.1, verbose=True):
    torch.manual_seed(SEED)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    x_train, y_train = train_set
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train))          # 매 epoch마다 순서를 섞는다
        total = 0.0
        for start in range(0, len(order), batch_size):  # batch 단위로 잘라서
            idx = order[start:start + batch_size]
            logits = model(x_train[idx])                # 1) 예측 (forward)
            loss = F.cross_entropy(logits, y_train[idx])  # 2) 얼마나 틀렸는지 (loss)
            optimizer.zero_grad()                       # 3) 이전 기울기 지우기
            loss.backward()                             # 4) 기울기 계산 (backward)
            optimizer.step()                            # 5) 가중치 수정
            total += loss.item() * len(idx)
        train_acc = accuracy(model, train_set)
        valid_acc = accuracy(model, valid_set)
        if verbose and (epoch <= 3 or epoch % 5 == 0):
            print(f"  epoch {epoch:2d}  loss {total / len(x_train):.4f}  train {train_acc:.3f}  valid {valid_acc:.3f}")
    return model


@torch.no_grad()
def accuracy(model, data):
    model.eval()
    x, y = data
    return (model(x).argmax(1) == y).float().mean().item()


def main():
    x, y = make_data()
    train_set, valid_set, test_set = split(x, y)
    print(f"데이터: train {len(train_set[0])}, validation {len(valid_set[0])}, test {len(test_set[0])}")

    print("\n[1] 선형 분류기 학습 (입력 2개 → 점수 2개)")
    torch.manual_seed(SEED)
    model = torch.nn.Linear(2, 2)
    train(model, train_set, valid_set)

    with torch.no_grad():
        logits = model(test_set[0])
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(1)
    print("\n[2] test 평가")
    print(f"  첫 샘플 logits {logits[0].numpy().round(3)} → softmax 확률 {probs[0].numpy().round(3)} → 예측 {pred[0].item()}")
    print(f"  정확도 {(pred == test_set[1]).float().mean().item():.3f}")
    print(f"  혼동행렬 [[정상→정상, 정상→고장], [고장→정상, 고장→고장]]\n{confusion(pred, test_set[1])}")

    print("\n[3] 오버피팅 체험: 정답을 무작위로 섞으면 배울 규칙이 없다")
    g = torch.Generator().manual_seed(SEED)
    y_random = y[torch.randperm(len(y), generator=g)]
    small_train = (x[:60], y_random[:60])                  # 작은 데이터 + 큰 모델
    random_valid = (x[420:510], y_random[420:510])
    torch.manual_seed(SEED)
    big = torch.nn.Sequential(torch.nn.Linear(2, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                              torch.nn.ReLU(), torch.nn.Linear(256, 2))
    optimizer = torch.optim.Adam(big.parameters(), lr=1e-2)
    for _ in range(1500):
        loss = F.cross_entropy(big(small_train[0]), small_train[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"  train 정확도 {accuracy(big, small_train):.3f}  ← 외워서 높다")
    print(f"  valid 정확도 {accuracy(big, random_valid):.3f}  ← 처음 보는 데이터는 찍는 수준(≈0.5)")


if __name__ == "__main__":
    main()
