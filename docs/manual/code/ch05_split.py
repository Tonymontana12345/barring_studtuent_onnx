"""5장 완성 코드: 전처리된 파일 목록을 train / validation / test로 나눈 split JSON을 만든다.

두 가지 방식 (src/bearing_kd/data.py의 create_split과 같은 규칙):
    1) 녹음 파일 단위 70/15/15  : 같은 베어링의 다른 파일이 train과 test에 함께 들어간다.
    2) 베어링 단위 holdout      : --test-bearings 로 지정한 베어링은 통째로 test.
                                  나머지 베어링의 파일을 85/15로 train/validation에 나눈다.

실행 (프로젝트 루트에서, 4장 맛보기 결과 사용):
    python docs/manual/code/ch05_split.py --test-bearings K002 KA04
    python docs/manual/code/ch05_split.py --output outputs/manual/split_file.json   # 파일 단위
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def bearing_label(code):
    if re.fullmatch(r"K\d{3}", code):
        return 0
    if re.fullmatch(r"K[ABI]\d{2}", code):
        return 1
    raise ValueError(f"알 수 없는 베어링 코드: {code}")


def create_split(paths, test_bearings=None, seed=42):
    rng = random.Random(seed)            # seed가 같으면 언제나 같은 분할 → 재현 가능
    groups = {"train": [], "validation": [], "test": []}
    if test_bearings:
        groups["test"] = [p for p in paths if p.parent.name in test_bearings]
        paths = [p for p in paths if p.parent.name not in test_bearings]
        if {bearing_label(p.parent.name) for p in groups["test"]} != {0, 1}:
            raise ValueError("test 베어링에 정상과 고장이 모두 있어야 합니다 (한쪽만 있으면 오판율을 잴 수 없음)")
    for label in (0, 1):                 # 정상/고장 비율이 각 세트에 고르게 가도록 라벨별로 나눈다
        subset = [p for p in paths if bearing_label(p.parent.name) == label]
        if len(subset) < 2:
            raise ValueError(f"라벨 {label}의 파일이 너무 적습니다: {len(subset)}개")
        rng.shuffle(subset)
        if test_bearings:
            a = max(1, int(len(subset) * .85))
            parts = (subset[:a], subset[a:], [])
        else:
            a, b = int(len(subset) * .7), int(len(subset) * .15)
            parts = (subset[:a], subset[a:a + b], subset[a + b:])
        for name, selected in zip(groups, parts):
            groups[name].extend(selected)
    for name in groups:
        rng.shuffle(groups[name])
        groups[name] = [str(p.relative_to(ROOT)) for p in groups[name]]  # 루트 기준 상대 경로로 저장
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="outputs/manual/processed")
    parser.add_argument("--output", default="outputs/manual/split_holdout.json")
    parser.add_argument("--test-bearings", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = sorted((ROOT / args.processed_dir).rglob("*_windows.npy"))
    if not paths:
        raise SystemExit("전처리된 파일이 없습니다. 4장 스크립트를 먼저 실행하세요.")
    groups = create_split(paths, args.test_bearings, args.seed)

    # 같은 파일이 두 세트에 들어가면 안 된다 (데이터 누수)
    all_files = sum(groups.values(), [])
    assert len(all_files) == len(set(all_files)), "중복 파일 발견"

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(groups, indent=2) + "\n")
    for name, files in groups.items():
        per_bearing = Counter(Path(f).parent.name for f in files)
        print(f"{name:10s} 파일 {len(files):3d}개  {dict(sorted(per_bearing.items()))}")
    print(f"\n저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
