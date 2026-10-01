"""0장 완성 코드: 매뉴얼을 따라갈 준비가 되었는지 프로젝트 폴더를 점검한다.

실행 (프로젝트 루트에서):
    python docs/manual/code/ch00_check_project.py
"""

from pathlib import Path

# 이 파일 위치: <루트>/docs/manual/code/ch00_check_project.py → 3단계 위가 프로젝트 루트
ROOT = Path(__file__).resolve().parents[3]

# (경로, 설명, 준비하는 챕터)
CHECKS = [
    ("src/bearing_kd", "프로젝트 소스 코드", "기본 포함"),
    ("pyproject.toml", "패키지 설정 파일", "기본 포함"),
    ("data/raw/paderborn", "Paderborn 원본 데이터", "3장"),
    ("models/MOMENT-1-base", "Teacher용 사전학습 모델", "7장"),
    ("models/MOMENT-1-small", "Student용 사전학습 모델", "7장"),
]


def main():
    print("프로젝트 점검\n")
    missing = []
    for rel, description, chapter in CHECKS:
        exists = (ROOT / rel).exists()
        mark = "OK " if exists else "없음"
        print(f"[{mark}] {rel:24s} {description} ({chapter})")
        if not exists:
            missing.append(chapter)

    raw = ROOT / "data/raw/paderborn"
    if raw.exists():
        bearings = sorted(p.name for p in raw.iterdir() if p.is_dir() and any(p.glob("*.mat")))
        print(f"\n압축이 풀린 베어링 {len(bearings)}개: {', '.join(bearings)}")

    print("\n결과:", "모두 준비됨" if not missing else f"아직 준비할 것 있음 → {', '.join(sorted(set(missing)))} 참고")


if __name__ == "__main__":
    main()
