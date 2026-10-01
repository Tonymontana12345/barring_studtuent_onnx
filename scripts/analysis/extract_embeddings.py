from bearing_kd.config import PROJECT_ROOT

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from momentfm import MOMENTPipeline

from bearing_kd.data import (
    BearingWindowDataset,
    make_moment_chunks,
)


MODEL_DIR = (PROJECT_ROOT / "models/MOMENT-1-base")

DATA_DIR = (PROJECT_ROOT / "data/processed/windows_1khz")

OUTPUT_DIR = (PROJECT_ROOT / "outputs/embeddings")

CONDITION = "N09_M07_F10"

BEARING_CODES = [
    "K001",
    "KA04",
    "KB24",
]

BATCH_SIZE = 16
MOMENT_LENGTH = 512


def find_window_paths(
    bearing_code,
):
    pattern = (
        f"{CONDITION}_"
        f"{bearing_code}_*_windows.npy"
    )

    paths = sorted(
        (
            DATA_DIR
            / bearing_code
        ).glob(pattern)
    )

    if not paths:
        raise FileNotFoundError(
            f"파일을 찾지 못했습니다: "
            f"{bearing_code}/{pattern}"
        )

    return paths


def create_embedding_model(
    device,
):
    print("[MOMENT-1-base Embedding 모델 로드]")

    model = MOMENTPipeline.from_pretrained(
        str(MODEL_DIR),
        model_kwargs={
            "task_name": "embedding",
            "n_channels": 1,
        },
        local_files_only=True,
    )

    model.init()
    model.to(device)
    model.eval()

    return model


def extract_embeddings(
    model,
    dataset,
    device,
):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # (B, 1, 4000)
            # → (B*8, 1, 512)
            moment_x, input_mask, _ = (
                make_moment_chunks(
                    batch_x,
                    batch_y,
                    moment_length=MOMENT_LENGTH,
                )
            )

            moment_x = moment_x.to(device)
            input_mask = input_mask.to(device)

            outputs = model(
                x_enc=moment_x,
                input_mask=input_mask,
            )

            chunk_embeddings = (
                outputs.embeddings
            )

            batch_size = batch_x.shape[0]

            chunk_count = (
                chunk_embeddings.shape[0]
                // batch_size
            )

            # Chunk Embedding:
            # (B*8, D)
            #
            # Window Embedding:
            # (B, 8, D)
            window_embeddings = (
                chunk_embeddings.reshape(
                    batch_size,
                    chunk_count,
                    -1,
                )
            )

            # 8개 Chunk Embedding을 평균
            window_embeddings = (
                window_embeddings.mean(dim=1)
            )

            all_embeddings.append(
                window_embeddings.cpu()
            )

            all_labels.append(
                batch_y.cpu()
            )

    embeddings = torch.cat(
        all_embeddings,
        dim=0,
    ).numpy()

    labels = torch.cat(
        all_labels,
        dim=0,
    ).numpy()

    return embeddings, labels


def main():
    device = torch.device("cpu")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = create_embedding_model(
        device
    )

    extracted = {}

    for bearing_code in BEARING_CODES:
        print()
        print(
            f"[{bearing_code} Embedding 추출]"
        )

        window_paths = find_window_paths(
            bearing_code
        )

        dataset = BearingWindowDataset(
            windows_paths=window_paths,
        )

        embeddings, labels = (
            extract_embeddings(
                model,
                dataset,
                device,
            )
        )

        extracted[bearing_code] = (
            embeddings,
            labels,
        )

        output_path = (
            OUTPUT_DIR
            / f"{bearing_code}_embeddings.npz"
        )

        np.savez_compressed(
            output_path,
            embeddings=embeddings,
            labels=labels,
        )

        print(
            f"Window 수: {len(dataset)}"
        )

        print(
            f"Embedding shape: "
            f"{embeddings.shape}"
        )

        print(
            f"저장: {output_path}"
        )

    # K001 정상 Embedding을 기준으로 중심 계산
    normal_embeddings = extracted[
        "K001"
    ][0]

    normal_center = (
        normal_embeddings.mean(
            axis=0
        )
    )

    print()
    print(
        "[정상 중심과의 Embedding 거리]"
    )

    for bearing_code in BEARING_CODES:
        embeddings = extracted[
            bearing_code
        ][0]

        distances = np.linalg.norm(
            embeddings - normal_center,
            axis=1,
        )

        print()
        print(bearing_code)
        print(
            f"최소 거리: "
            f"{distances.min():.6f}"
        )
        print(
            f"최대 거리: "
            f"{distances.max():.6f}"
        )
        print(
            f"평균 거리: "
            f"{distances.mean():.6f}"
        )


if __name__ == "__main__":
    main()
