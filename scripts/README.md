# Scripts

## Overview

| Script | Purpose | When to run |
|--------|---------|-------------|
| `create_mock_data.py` | Generate small fake HDF5 files for local development | Once, before downloading real data |
| `compute_stats.py` | Compute per-channel and label normalization stats from training data | Once, after downloading real data |
| `test_dataset.py` | Validate the full data pipeline end-to-end | After any change to `src/dataset.py` or data |
| `train.slurm` | Slurm job script to launch GPU training | When submitting training jobs |

---

## create_mock_data.py

Generates 10 small HDF5 files in `data/mock/` with the same structure as the real data (5 alpha values × 2 zeta values). Each file has 2 trajectories with 20 timesteps at 256×256 resolution. Use this to develop and test code without needing the full 52GB dataset.

    python scripts/create_mock_data.py

Output: `data/mock/*.hdf5` (gitignored)

---

## compute_stats.py

Scans all training HDF5 files and computes:
- **Channel stats**: per-channel mean and std across all 11 physical channels (used for z-score normalization)
- **Label stats**: mean and std of alpha and zeta values (used for z-score normalization of regression targets)

These stats are saved to a JSON file and loaded by `src/dataset.py` at runtime.

    python scripts/compute_stats.py --root /scratch/$USER/data/active_matter --split train --output data/stats/train_stats.json

Arguments:
- `--root` — path to the dataset root (contains `data/train/`)
- `--split` — must be `train` (computing stats on val/test would be data leakage)
- `--output` — where to save the JSON (default: `data/stats/train_stats.json`)
- `--crop-size` — spatial crop size, `224` to match model input, `0` for no crop
- `--time-chunk` — timesteps loaded per batch, controls memory usage (default: 8)

Output: `data/stats/train_stats.json`

---

## test_dataset.py

Runs 8 checks on the data pipeline to verify everything works:

1. All splits (train/valid/test) load correctly with expected trajectory counts
2. SSL mode returns tensors only (no label leakage)
3. Eval mode returns (frames, labels) with correct shapes
4. Channel normalization produces means ≈ 0 and stds ≈ 1, with per-channel breakdown
5. Val/test samples are deterministic (no random augmentation)
6. DataLoader batching produces correct shapes, no NaN/Inf
7. `build_dataloaders` factory works end-to-end
8. Label distribution matches expected alpha/zeta values

Output is printed to terminal and saved to a log file.

    python scripts/test_dataset.py --root /scratch/$USER/data/active_matter --stats data/stats/train_stats.json

Arguments:
- `--root` — path to the dataset root
- `--stats` — path to pre-computed stats JSON (default: `data/stats/train_stats.json`)
- `--output` — path to save the log file (default: `data/logs/test_dataset.log`)

Output: `data/logs/test_dataset.log`

---

## train.slurm

Slurm batch script for submitting GPU training jobs on the NYU HPC cluster.

    sbatch scripts/train.slurm

Configuration:
- Partition: `c12m85-a100-1` (1× A100 40GB)
- Time limit: 24 hours
- Memory: 85GB
- Includes `--requeue` for automatic restart on spot instance preemption
- Logs go to `/scratch/$USER/logs/`