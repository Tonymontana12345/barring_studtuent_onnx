"""Download one weight format per model, pinning the resolved repository commit."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from bearing_kd.config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--base-revision", default="main")
    parser.add_argument("--small-revision", default="main")
    args = parser.parse_args()
    for size in ("base", "small"):
        folder = args.output_dir / f"MOMENT-1-{size}"
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Preserve existing model files; choose a new --output-dir: {folder}")
    manifest = {}
    for size, revision in [("base", args.base_revision), ("small", args.small_revision)]:
        repo = f"AutonLab/MOMENT-1-{size}"
        commit = HfApi().model_info(repo, revision=revision).sha
        snapshot_download(repo_id=repo, revision=commit, local_dir=args.output_dir / f"MOMENT-1-{size}",
                          allow_patterns=["config.json", "model.safetensors", "README.md", "LICENSE*"])
        manifest[repo] = {"revision": commit}
    (args.output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
