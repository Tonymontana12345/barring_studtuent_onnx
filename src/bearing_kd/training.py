"""Teacher LoRA training, verified prediction caches, and head-only distillation."""

import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import BearingWindowDataset, DistillationDataset, load_split_paths, make_moment_chunks, sha256_file
from .models import LORA_CONFIG, create_model, load_model, window_logits


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def kd_loss(student_logits, teacher_logits, labels, temperature=2., hard_weight=.5):
    if temperature <= 0 or not 0 <= hard_weight <= 1:
        raise ValueError("Temperature must be positive and hard_weight must be in [0, 1]")
    hard = F.cross_entropy(student_logits, labels)
    soft = F.kl_div(F.log_softmax(student_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1), reduction="batchmean") * temperature**2
    return hard_weight * hard + (1 - hard_weight) * soft


def validate_gradients(model):
    """Fail rather than silently training only part of the intended parameter set."""
    # 기존 Teacher는 LoRA에 기울기가 전달되지 않았는데도(grad=None) 에러 없이 학습이 끝났다.
    # optimizer는 grad=None 파라미터를 조용히 건너뛰기 때문이다. 원인과 수정은
    # models.create_model()의 주석 참고. 같은 문제가 재발하면 첫 step에서 여기서 멈춘다.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise RuntimeError(f"Missing gradient for trainable parameter: {name}")
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Non-finite gradient: {name}")


def predict(model, dataset, device="cpu", batch_size=16):
    model.eval()
    logits = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits.append(window_logits(model, batch[0].to(device)).cpu().numpy())
    return np.concatenate(logits)


def metrics(logits, labels):
    logits, labels = np.asarray(logits), np.asarray(labels)
    if logits.shape != (len(labels), 2) or not len(labels) or not np.isfinite(logits).all():
        raise ValueError("Invalid predictions for evaluation")
    predictions = logits.argmax(axis=1)
    confusion = np.zeros((2, 2), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    return {"correct": int((predictions == labels).sum()), "total": len(labels),
            "accuracy": float((predictions == labels).mean()), "confusion_matrix": confusion.tolist()}


def evaluate(model_dir, checkpoint, split_path, root, teacher=False, device="cpu", batch_size=16,
             processed_dir=None):
    paths = sorted(Path(processed_dir).rglob("*_windows.npy")) if processed_dir else load_split_paths(split_path, root)[2]
    dataset = BearingWindowDataset(paths, root)
    model = load_model(model_dir, checkpoint, device, teacher)
    result = metrics(predict(model, dataset, device, batch_size), dataset.labels)
    result["dataset_fingerprint"] = dataset.fingerprint
    return result


def save_cache(path, dataset, logits, checkpoint):
    logits = np.asarray(logits, dtype=np.float32)
    metrics(logits, dataset.labels)  # Validate shape and finiteness before saving.
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    np.savez_compressed(
        path, logits=logits, labels=dataset.labels, predictions=logits.argmax(1),
        probabilities=probabilities, sample_ids=dataset.sample_ids,
        dataset_fingerprint=np.asarray(dataset.fingerprint),
        teacher_sha256=np.asarray(sha256_file(checkpoint)),
    )


def cache_teacher(model_dir, checkpoint, split_path, output_dir, root, device="cpu", batch_size=16):
    train, valid, _ = load_split_paths(split_path, root)
    output_dir = Path(output_dir)
    if any((output_dir / f"teacher_{name}.npz").exists() for name in ("train", "validation")):
        raise FileExistsError("Use a new cache directory to preserve existing teacher predictions")
    datasets = [BearingWindowDataset(paths, root) for paths in (train, valid)]
    model = load_model(model_dir, checkpoint, device, teacher=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, dataset in zip(("train", "validation"), datasets):
        logits = predict(model, dataset, device, batch_size)
        save_cache(output_dir / f"teacher_{name}.npz", dataset, logits, checkpoint)
        result[name] = metrics(logits, dataset.labels)
        print(f"Cached {name}: {len(dataset)} windows", flush=True)
    return result


def train(model_dir, split_path, output_dir, root, teacher=False, cache_dir=None,
          epochs=20, batch_size=16, learning_rate=1e-4, seed=42, device="cpu"):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    output_dir = Path(output_dir)
    filename = "teacher_lora_best.pt" if teacher else "student_kd_best.pt"
    if (output_dir / filename).exists():
        raise FileExistsError(f"Use a new output directory: {output_dir / filename}")
    seed_everything(seed)
    train_paths, valid_paths, _ = load_split_paths(split_path, root)
    if teacher:
        datasets = [BearingWindowDataset(paths, root) for paths in (train_paths, valid_paths)]
    else:
        if cache_dir is None:
            raise ValueError("Student training requires --cache-dir")
        datasets = [DistillationDataset(paths, Path(cache_dir) / f"teacher_{name}.npz", root)
                    for name, paths in zip(("train", "validation"), (train_paths, valid_paths))]
        if datasets[0].teacher_sha256 != datasets[1].teacher_sha256:
            raise ValueError("Train/validation caches were generated by different teachers")
    model = create_model(model_dir, teacher=teacher).to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    config = {"seed": seed, "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "training_scope": "lora-and-head" if teacher else "head-only",
              "temperature": 2., "hard_loss_weight": .5,
              "split_sha256": sha256_file(split_path),
              "trainable_parameters": sum(p.numel() for p in parameters),
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "momentfm", "peft", "numpy")},
              "dataset_fingerprints": [
                  (d if teacher else d.base_dataset).fingerprint for d in datasets]}
    if not teacher:
        config["teacher_sha256"] = datasets[0].teacher_sha256
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)
    loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    valid_data = datasets[1] if teacher else datasets[1].base_dataset
    best, history = -1., []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total = 0., 0
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            if teacher:
                chunks, mask, chunk_y = make_moment_chunks(x, y)
                loss = F.cross_entropy(model(x_enc=chunks, input_mask=mask).logits, chunk_y)
            else:
                loss = kd_loss(window_logits(model, x), batch[2].to(device), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            validate_gradients(model)
            optimizer.step()
            total_loss += loss.item() * len(y)
            total += len(y)
        result = metrics(predict(model, valid_data, device, batch_size), valid_data.labels)
        record = {"epoch": epoch, "train_loss": total_loss / total, "validation": result}
        history.append(record)
        print(json.dumps(record), flush=True)
        if result["accuracy"] >= best:
            best = result["accuracy"]
            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "validation_accuracy": best, "training_config": config}
            if teacher:
                checkpoint["lora_config"] = LORA_CONFIG
            torch.save(checkpoint, output_dir / filename)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {"checkpoint": str(output_dir / filename), "best_validation_accuracy": best}
