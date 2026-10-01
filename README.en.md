# Bearing KD ONNX

[한국어](README.md) | **English**

An experimental project for classifying normal/faulty bearings from Paderborn vibration data. A Teacher (MOMENT Base with LoRA and a classification head) is used to distill **only the classification head** of MOMENT Small, which is then exported to ONNX for Raspberry Pi CPU inference.

> **Current status (2026-09-25)**
> - The deployment path (ONNX export → Raspberry Pi loading → numerical parity with Mac → real-time speed) is **verified**.
> - Model performance is **not yet deployment-ready.** On bearings unseen during training, the Student reaches 64.6% and detects **0%** of KB24 faults.
> - Validation built from different files of the same bearings (98–100%) greatly overstates real performance. See [v3 experiment](#v3-experiment-bearing-level-holdout).

## Layout

```text
src/bearing_kd/
  preprocessing.py  # shared preprocessing and labels for all bearings
  data.py           # file splits, Dataset, chunking, cache integrity
  models.py         # MOMENT/LoRA, strict checkpoint loading
  training.py       # Teacher training, prediction cache, Student KD, evaluation
  export.py         # ONNX export + numerical comparison against PyTorch
  inference.py      # window-level ONNX inference without PyTorch
  cli.py            # bearing-kd command
scripts/analysis/   # waveform, spectrum and embedding analysis
scripts/verify_pipeline.py # integration check on real data/models
scripts/download_models.py # fetch pretrained models
scripts/pi_verify.py       # Raspberry Pi ONNX parity check (numpy + onnxruntime only)
tests/             # automated regression tests, no downloads needed
data/splits/       # file lists for reproducibility (tracked in Git)
docs/              # architecture, legacy code mapping, verification log (Korean)
```

Raw/processed data, model weights, outputs and virtual environments are excluded by `.gitignore`.

## Installation

Uses Python **3.11** and `uv`. MOMENT 0.1.4 pins old numpy/transformers/huggingface-hub versions that conflict with current PEFT, so the tested versions are listed in `constraints-tested.txt` and installed with uv overrides. A plain `pip install -r requirements.txt` cannot resolve this conflict.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-lock.txt --override constraints-tested.txt
bearing-kd --help
```

For ONNX-only inference (e.g. Raspberry Pi), PyTorch/MOMENT are not required:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
```

> **Raspberry Pi note:** the verified Pi environment is a Python **3.13** virtualenv (`~/vib-onnx/.venv`). `pyproject.toml` declares `requires-python = ">=3.11,<3.12"`, so `pip install -e .` (`bearing-kd infer`) refuses to install there. For now use a standalone numpy + onnxruntime script such as `scripts/pi_verify.py`. Running without activating the venv gives `No module named 'onnxruntime'`; call `.venv/bin/python` directly to avoid that.

Versions were recorded on macOS ARM64; wheel availability may differ on other platforms. Because of MOMENT's original metadata, `pip check` may report the three overridden packages as conflicts.

## Data and models

Download the data from the [official Paderborn Bearing DataCenter](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter) and extract it as `data/raw/paderborn/K001/*.mat`. Bearing codes are labeled K + 3 digits = normal (0), KA/KB/KI = fault (1). After adding data, re-run splitting, caching and training.

`.rar` archives can be extracted with macOS's built-in `bsdtar` (`bsdtar -xf KA05.rar`); one bearing is about 670 MB extracted.

Pretrained models: [MOMENT Base](https://huggingface.co/AutonLab/MOMENT-1-base), [MOMENT Small](https://huggingface.co/AutonLab/MOMENT-1-small).

```bash
python scripts/download_models.py
```

The script fetches `config.json`, `model.safetensors` and the model card, and records the resolved commits in `models/download_manifest.json` (`--base-revision`, `--small-revision` pin specific commits).

**Trained checkpoints are not in Git.** To evaluate from a fresh clone, obtain checkpoints separately or run the training steps below.

## Preprocessing contract

| Item | Value |
|---|---|
| Channel | `vibration_1`, single axis |
| Raw sampling rate | 64,000 Hz |
| Low-pass | 1,000 Hz, 8th-order Butterworth, `sosfiltfilt` |
| Resampling | 4,000 Hz |
| Window / stride | 4,000 / 2,000 samples |
| MOMENT chunks | 8 × 512 samples |
| Padding | last 96 samples are 0 with mask 0 |
| Labels | K + 3 digits: normal 0 / KA·KB·KI: fault 1 |
| Window decision | **mean of the 8 chunk logits**, then argmax |

MOMENT contains RevIN internally, so no extra window normalization is applied. The zero-phase filter processes whole recordings; a stateful streaming filter for real-time use is not included.

### Why 4 kHz, and what it costs

64 kHz is the **sampling rate** (samples recorded per second), not a signal frequency. Recording at 64 kHz captures 0–32 kHz. Resampling to 4 kHz limits the highest representable frequency to **2 kHz** (Nyquist). Faster vibrations oscillate between samples and cannot be recovered; without filtering they fold into fake low frequencies (aliasing). Hence the 1 kHz low-pass before resampling.

MOMENT's input length is fixed at 512 samples, so the sampling rate decides both the time span and the frequency band the model sees.

| Sampling | 512 samples = | Visible band | Pi cost (chunks per window) |
|---|---|---|---|
| 64 kHz | 8 ms | 0–32 kHz | 125 |
| 16 kHz | 32 ms | 0–8 kHz | 32 |
| **4 kHz (current)** | **128 ms** | **0–2 kHz** | **8** |

4 kHz was chosen for **Raspberry Pi inference speed**, at the cost of discarding the high-frequency band that carries much of the fault information. Since the field sensor may sample at only a few kHz, the current direction is to **give up the high band and improve generalization within 0–1 kHz** (bearing-level validation, more training bearings).

## Running a new experiment

Every command that creates files requires an explicit output location; use new paths so results never mix. Training takes time.

```bash
# 1. Shared preprocessing
bearing-kd preprocess --bearings K001 KA03 KA04 --output-dir data/processed/windows_v2
# 2. Recording-level 70/15/15 split
bearing-kd split --processed-dir data/processed/windows_v2 --output data/splits/file_split_v2.json
# 3. Teacher LoRA + classification head
bearing-kd train-teacher --split data/splits/file_split_v2.json --output-dir outputs/runs/v2/teacher
# 4. Teacher prediction cache
bearing-kd cache-teacher --split data/splits/file_split_v2.json --checkpoint outputs/runs/v2/teacher/teacher_lora_best.pt --output-dir outputs/runs/v2/cache
# 5. Student head distillation
bearing-kd train-student --split data/splits/file_split_v2.json --cache-dir outputs/runs/v2/cache --output-dir outputs/runs/v2/student
# 6. Test evaluation
bearing-kd evaluate-student --split data/splits/file_split_v2.json --checkpoint outputs/runs/v2/student/student_kd_best.pt
# 7. ONNX export + parity check
bearing-kd export --checkpoint outputs/runs/v2/student/student_kd_best.pt --output outputs/runs/v2/onnx/student.onnx
```

**Bearing-level holdout** is set with `--test-bearings`: every recording of those bearings goes to test, and the remaining bearings are split 85/15 into train/validation by recording. The split is rejected unless the test set contains both normal and fault bearings, since a fault-only test set would give 100% to a model that always answers "fault".

```bash
bearing-kd split --processed-dir data/processed/windows_v3 --test-bearings K004 KB24 KB27 --output data/splits/bearing_holdout_v3.json
```

Defaults: CPU, 20 epochs, batch 16, learning rate 1e-4, seed 42. Change them with `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--device cuda|mps`.

**Apple Silicon:** CUDA is NVIDIA-only; on a Mac use `--device mps` (Metal). Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to run unsupported ops on CPU. Measured Teacher training speed (batch 16): 6.4 s on CPU vs 3.1 s on MPS. Results are not bit-identical to CPU even with the same seed.

**Training time:** after the LoRA fix (below), backpropagation passes through the whole encoder and gradient checkpointing recomputes the forward pass, making training about 3–4× slower than before. The v3 Teacher (~2,860 windows) took ~10.7 min/epoch on MPS. The earlier "20 epochs in one hour" was fast only because LoRA was not actually being trained.

**Checkpoint caveat:** the code overwrites `*_best.pt` when validation accuracy is **greater than or equal to** the best so far. Once validation saturates at 100%, only the latest 100% epoch survives (in v3, epoch 4 was overwritten by epochs 9 and 10). Per-epoch checkpoints are not implemented yet.

Student KD uses temperature 2, hard-label weight 0.5, and a frozen backbone. Configuration, data hashes and per-epoch results are written to the training folder. Results may differ across hardware and library versions even with the same seed.

Paths are resolved relative to the project root. For other data locations use `bearing-kd --root /path/to/workspace ...`. The global options `--root` and `--threads` go **before** the subcommand.

## Evaluation and inference

```bash
bearing-kd evaluate-student
bearing-kd evaluate-teacher
bearing-kd evaluate-teacher --processed-dir data/processed/windows_1khz/KB24
bearing-kd infer --input data/processed/windows_1khz/K001/N09_M07_F10_K001_1_windows.npy
bearing-kd infer --input data/raw/paderborn/K001/N09_M07_F10_K001_1.mat
```

`infer` returns logits and a label for each 1-second window. NPY input must be preprocessed `[N,4000]` or `[4000]`; MAT input goes through the shared preprocessing. `.onnx.data` holds the external weights and must sit next to the `.onnx` file.

## Evaluation scope and legacy results

**LoRA bug:** all 24 LoRA B matrices of the original Teacher were zero. With MOMENT's default reentrant gradient checkpointing, a frozen embedder meant the encoder input did not require gradients, so no gradient reached the LoRA parameters and only the head was trained. The code now uses non-reentrant checkpointing and fails fast if any trainable parameter is missing a gradient (see comments in `models.create_model()`).

The legacy Student reproduced **251/252 (99.60%)** on its test windows, confusion matrix `[[83,1],[0,168]]`. That split divides different recordings of the same K001·KA03·KA04 bearings, so it is **not an evaluation of generalization to new bearings or field sensors.**

On KB24 the legacy Teacher and Student both score **0/560 (0%)**, predicting everything as normal. Re-preprocessing KB24 with the 8th-order filter gives the same result, so the filter order is not the cause. KI03, for comparison, scored 70/70.

Legacy KB data used a 6th-order filter; the new shared preprocessing uses 8th order for all bearings. Do not mix legacy model results with results from the new preprocessing/retraining. Legacy Teacher NPZ caches have no sample IDs/hashes and are rejected by the new trainer; regenerate them with `cache-teacher`.

## v3 experiment: bearing-level holdout

Run because the legacy evaluation (recording-level split, 99.6%) could not show performance on new bearings. Results are in `outputs/runs/v3/`.

| Setup | Details |
|---|---|
| Preprocessing | `data/processed/windows_v3` (9 bearings, shared 8th-order preprocessing, 720 files / 5,042 windows; one K002 and one KB27 recording are longer and yield 8 windows) |
| Train (408 files) / Validation (72 files) | K001·K002·K003 + KA03·KA04·KI03, recording-level 85/15 |
| **Test (240 files)** | **K004 (normal) + KB24·KB27 (fault)**, fully excluded from training |
| Teacher | MOMENT Base + LoRA on MPS; planned 20 epochs, **stopped at epoch 10** (validation hit 100%) |
| Student | MOMENT Small head KD, 20 epochs, best validation 98.0% (epoch 20) |

| Test bearing | Teacher (epoch 10) | Student (PyTorch = ONNX) |
|---|---|---|
| K004 normal | 420/560 (75.0%) | **525/560 (93.8%)** |
| KB24 fault | **0/560 (0%)** | **0/560 (0%)** |
| KB27 fault | 561/561 (100%) | 561/561 (100%) |
| Overall | 981/1681 (**58.4%**) | 1086/1681 (**64.6%**) |
| Confusion matrix | `[[420,140],[560,561]]` | `[[525,35],[560,561]]` |

- Validation (same bearings, different files) was 99.4–100% for the Teacher and 96.6–98.0% for the Student, yet test was 58–65%: **the models overfit to individual bearings.**
- Adding KI03 to training left KB24 at 0%; KB27 is classified perfectly.
- K004 errors vary strongly by operating condition (Teacher: N15_M01_F10 71/140, N15_M07_F10 83/140, N09_M07_F10 128/140, N15_M07_F04 138/140).
- The Student raises fewer false alarms (140 → 35), plausibly because a head-only model memorizes less, but with a single normal test bearing this is not conclusive.
- All 24/24 LoRA B matrices of the Teacher trained with the fixed code were actually updated.

Reproduce:

```bash
bearing-kd preprocess --bearings K001 K002 K003 K004 KA03 KA04 KI03 KB24 KB27 --output-dir data/processed/windows_v3
bearing-kd split --processed-dir data/processed/windows_v3 --test-bearings K004 KB24 KB27 --output data/splits/bearing_holdout_v3.json
# Run scripts: outputs/runs/v3/run.sh (Teacher), outputs/runs/v3/run_student.sh (cache → KD → evaluation → export)
```

## Raspberry Pi deployment check

| Item | Result |
|---|---|
| Loading | OK, CPUExecutionProvider |
| I/O | `x_enc [1,1,512]`, `input_mask [1,512]` → `logits [1,2]`, all float32 |
| Parity (K004·KB27, one file per operating condition, 56 windows) | **PASS**: max difference vs Mac 7.2e-7, predictions 56/56 identical, correct 55/56 |
| Speed | **38.9 ms per chunk, 311 ms per window** (Mac CPU, 2 threads: 3.2 ms/chunk) |

Windows advance every 0.5 s, so anything under 500 ms per window keeps up in real time; the current model uses about 60% of that budget.

```bash
# Copy Mac → Pi (.onnx and .onnx.data must be in the same folder; do not rename)
scp -r outputs/runs/v3/pi_verify <user>@<pi-address>:~/vib-onnx/
# Run on the Pi
cd ~/vib-onnx/pi_verify && ~/vib-onnx/.venv/bin/python pi_verify.py
```

- `pi_verify_data.npz` is a NumPy archive holding the preprocessed windows, true labels, the logits expected from the Mac run, and source names. It can live anywhere; pass it with `--data`.
- The `.onnx` file stores the **file name** of its `.onnx.data`: `student.onnx.data` for the new model, `student_moment_chunk.onnx.data` for the legacy one.
- **Not yet verified:** preprocessing (filter and resampling) applied on the Pi directly to raw sensor data, and a real-time streaming filter.

## Known issues and next steps

| Area | Details |
|---|---|
| Performance | KB24 detected at 0%; poor generalization to new bearings |
| Evaluation | Validation uses other files of the same bearings, so it is useless for model selection; bearing-level validation (`--val-bearings`) is needed |
| Training | No per-epoch checkpoints; `training_config.json` does not record `model_dir` |
| Decisions pending | **Field sensor sampling rate** (sets preprocessing direction), which additional bearings to use |
| Ablations | With vs without LoRA; KD vs a Small head trained on hard labels only. The actual benefit of distillation is unverified |
| Install | The Pi runs Python 3.13 but `requires-python` is `<3.12` |

## Tests

```bash
# No data or model downloads needed
python -m unittest discover -s tests -v
# Needs local raw data/models: full test evaluation + fresh export + small real training run
HF_HUB_OFFLINE=1 python scripts/verify_pipeline.py
```

There are 18 regression tests, including the bearing-holdout split. Integration results are saved in `outputs/verification/run-*`. The training smoke test uses one window per class for one epoch and checks wiring only, not performance.

## Attribution and license

Data: Christian Lessmeier et al., Chair of Design and Drive Technology, Paderborn University. The [official data page](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter) specifies CC BY-NC 4.0 with attribution. For MOMENT, see the respective model cards.

No license has been chosen for the project's own code; the repository owner must select one. Terms for external data and models apply separately.
