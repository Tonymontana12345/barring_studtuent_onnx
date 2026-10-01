"""16장 완성 코드: 허깅페이스 업로드 폴더를 준비하고, 공개 전 점검을 한다. (업로드는 --upload 를 줄 때만)

    1) 업로드 폴더 만들기: onnx/, verify/, README.md(모델 카드)
    2) 공개 전 점검: 파일 안에 컴퓨터 경로·사용자 이름·토큰처럼 보이는 문자열이 없는지
    3) (선택) 업로드: 먼저 터미널에서 `hf auth login` 으로 로그인해 두어야 한다

실행 (프로젝트 루트에서, 14·15장 결과 사용):
    python docs/manual/code/ch16_huggingface.py                         # 준비 + 점검만
    python docs/manual/code/ch16_huggingface.py --upload --repo-id 내계정/저장소이름 --private
"""

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

MODEL_CARD = """---
license: cc-by-nc-4.0
library_name: onnx
base_model: AutonLab/MOMENT-1-small
tags:
  - onnx
  - time-series-classification
  - bearing-fault-diagnosis
  - raspberry-pi
---

# {title}

> **Experimental model.** A vibration anomaly analysis model that runs on Raspberry Pi.
> Additional training may be required depending on the application environment.

## Files
- `onnx/student.onnx`, `onnx/student.onnx.data` — keep both in the same folder, do not rename.
- `verify/pi_infer.py`, `verify/pi_verify_data.npz` — parity check: `python verify/pi_infer.py verify`

## Input
Single-axis vibration (64 kHz) → 1 kHz low-pass (8th-order Butterworth, zero-phase) → 4 kHz →
1-s windows (4000 samples, stride 2000) → 8 chunks × 512 (last 96 zero-padded, mask 0).
ONNX input `x_enc [1,1,512]`, `input_mask [1,512]` → `logits [1,2]`. Window decision = mean of 8 chunk logits.

## Attribution
Data: Paderborn University Bearing DataCenter (CC BY-NC 4.0). Base model: AutonLab/MOMENT-1-small.
"""

# 공개하면 안 되는 것처럼 보이는 문자열 패턴
PATTERNS = {
    "macOS 사용자 경로": re.compile(rb"/Users/[^/\s\"']+"),
    "Linux 사용자 경로": re.compile(rb"/home/[^/\s\"']+"),
    "Windows 사용자 경로": re.compile(rb"C:\\\\Users\\\\"),
    "허깅페이스 토큰": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "개인 키": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}


def prepare(onnx_path, bundle_dir, out_dir, title):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "onnx").mkdir(parents=True)
    (out_dir / "verify").mkdir()
    for f in (onnx_path, Path(str(onnx_path) + ".data")):
        shutil.copy2(f, out_dir / "onnx" / f.name)
    # 15장 스크립트는 같은 폴더의 student.onnx를 찾으므로, 업로드 구조(onnx/ 폴더)에 맞게 경로만 바꾼다
    script = (bundle_dir / "pi_infer.py").read_text().replace('HERE / "student.onnx"', 'HERE.parent / "onnx" / "student.onnx"')
    (out_dir / "verify" / "pi_infer.py").write_text(script)
    shutil.copy2(bundle_dir / "pi_verify_data.npz", out_dir / "verify" / "pi_verify_data.npz")
    (out_dir / "README.md").write_text(MODEL_CARD.format(title=title))


def scan(folder):
    """모든 파일을 바이트 단위로 훑어 위험한 문자열을 찾는다 (.onnx 안의 메타데이터 포함)."""
    findings = []
    for f in sorted(folder.rglob("*")):
        if f.is_file():
            data = f.read_bytes()
            for name, pattern in PATTERNS.items():
                hits = pattern.findall(data)
                if hits:
                    findings.append((f.relative_to(folder), name, len(hits)))
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="outputs/manual/onnx_clean/student.onnx")
    parser.add_argument("--bundle", default="outputs/manual/pi_bundle")
    parser.add_argument("--output-dir", default="outputs/manual/hf_upload")
    parser.add_argument("--title", default="MOMENT-Small Bearing Vibration Classifier (ONNX, manual build)")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo-id")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    out = ROOT / args.output_dir
    prepare(ROOT / args.onnx, ROOT / args.bundle, out, args.title)
    print(f"업로드 폴더: {out.relative_to(ROOT)}")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {str(f.relative_to(out)):28s} {f.stat().st_size / 1e6:6.1f} MB")

    findings = scan(out)
    print("\n공개 전 점검:", "문제 없음" if not findings else "")
    for path, name, n in findings:
        print(f"  ⚠ {path}: {name} {n}건")

    if args.upload:
        if findings:
            raise SystemExit("점검에서 문제가 발견되어 업로드하지 않습니다.")
        if not args.repo_id:
            raise SystemExit("--repo-id 계정/저장소이름 을 지정하세요.")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo_id, private=args.private, exist_ok=True)
        info = api.upload_folder(repo_id=args.repo_id, folder_path=str(out), commit_message="Upload manual build")
        print("업로드 완료:", info)


if __name__ == "__main__":
    main()
