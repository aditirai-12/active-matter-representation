#!/usr/bin/env python3
"""Quick validation of the data pipeline using pre-computed stats.

Usage (from repo root):
  python scripts/test_dataset.py --root /scratch/$USER/data/active_matter --stats data/stats/train_stats.json

Output is printed to terminal AND saved to data/logs/test_dataset.log (override with --output).
"""

import argparse
import io
import os
import sys
import time

import torch
from torch.utils.data import DataLoader


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()

# add repo root to path so we can import src/
sys.path.insert(0, ".")

from src.dataset import (
    ActiveMatterDataset,
    load_stats,
    build_dataloaders,
    N_FRAMES,
    N_CHANNELS,
    CROP_H,
)


def step_timer():
    """Returns a callable that prints elapsed time since creation."""
    start = time.perf_counter()
    def elapsed():
        return time.perf_counter() - start
    return elapsed


def log(msg, end="\n", flush=True):
    """Print with timestamp."""
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", end=end, flush=flush)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--stats", type=str, default="data/stats/train_stats.json")
    parser.add_argument("--output", type=str, default="data/logs/test_dataset.log",
                        help="Path to save test output log")
    args = parser.parse_args()

    tee = Tee(args.output)
    sys.stdout = tee

    total_start = time.perf_counter()

    print(f"[{time.strftime('%H:%M:%S')}] Starting dataset pipeline test")
    print(f"  root   : {args.root}")
    print(f"  stats  : {args.stats}")
    print(f"  output : {args.output}\n")

    log("Loading stats JSON... ", end="")
    channel_stats, label_stats = load_stats(args.stats)
    print(f"done  ({len(channel_stats['mean'])} channels, "
          f"alpha std={label_stats['alpha']['std']:.4f}, "
          f"zeta std={label_stats['zeta']['std']:.4f})")

    # ── 1. Check all splits load ──────────────────────────────────
    print("\n── 1. Split sizes " + "─" * 40)
    for split in ("train", "valid", "test"):
        log(f"Loading {split} split... ", end="")
        t = step_timer()
        try:
            ds = ActiveMatterDataset(
                args.root, split,
                channel_stats=channel_stats,
                label_stats=label_stats,
                return_labels=True,
            )
            alphas, zetas = ds.get_all_labels()
            print(f"done in {t():.1f}s")
            log(f"  {split:5s}: {len(ds):>4d} trajectories  |  "
                f"{len(ds.files)} files  |  "
                f"alpha unique: {len(set(alphas.tolist()))}  "
                f"zeta unique: {len(set(zetas.tolist()))}")
        except FileNotFoundError as e:
            print(f"NOT FOUND")
            log(f"  {e}")

    # ── 2. SSL mode (no labels) ──────────────────────────────────
    print("\n── 2. SSL mode — no labels " + "─" * 30)
    log("Creating SSL dataset (return_labels=False)... ", end="")
    t = step_timer()
    ds_ssl = ActiveMatterDataset(
        args.root, "train",
        channel_stats=channel_stats,
        label_stats=label_stats,
        return_labels=False,
    )
    print(f"done in {t():.1f}s")

    log("Loading first sample... ", end="")
    t0 = time.perf_counter()
    item = ds_ssl[0]
    ms = (time.perf_counter() - t0) * 1000
    print(f"done in {ms:.0f}ms")

    assert not isinstance(item, tuple), "FAIL: labels leaked in SSL mode!"
    assert item.shape == (N_FRAMES, N_CHANNELS, CROP_H, CROP_H), \
        f"FAIL: expected ({N_FRAMES},{N_CHANNELS},{CROP_H},{CROP_H}), got {tuple(item.shape)}"
    log(f"shape     : {tuple(item.shape)} ✓")
    log(f"dtype     : {item.dtype}")
    log(f"no labels : ✓  (returned {type(item).__name__}, not tuple)")

    # ── 3. Eval mode (with labels) ───────────────────────────────
    print("\n── 3. Eval mode — with labels " + "─" * 27)
    log("Creating eval dataset (return_labels=True)... ", end="")
    t = step_timer()
    ds_eval = ActiveMatterDataset(
        args.root, "train",
        channel_stats=channel_stats,
        label_stats=label_stats,
        return_labels=True,
    )
    print(f"done in {t():.1f}s")

    log("Loading first sample with labels... ", end="")
    t = step_timer()
    frames, labels = ds_eval[0]
    print(f"done in {t():.3f}s")

    assert frames.shape == (N_FRAMES, N_CHANNELS, CROP_H, CROP_H)
    assert labels.shape == (2,)
    log(f"frames : {tuple(frames.shape)} ✓")
    log(f"labels : [{labels[0]:.4f}, {labels[1]:.4f}]  [alpha_z, zeta_z] ✓")

    # ── 4. Channel normalization sanity check ────────────────────
    print("\n── 4. Channel normalization " + "─" * 29)
    n_samples = min(16, len(ds_eval))
    log(f"Loading {n_samples} samples for normalization check...")
    t = step_timer()
    samples = []
    for i in range(n_samples):
        samples.append(ds_eval[i][0])
        if (i + 1) % 4 == 0 or i == n_samples - 1:
            log(f"  loaded {i+1}/{n_samples} samples ({t():.1f}s elapsed)")

    batch = torch.stack(samples)
    ch_mean = batch.mean(dim=(0, 1, 3, 4))
    ch_std = batch.std(dim=(0, 1, 3, 4))

    channel_names = (
        ["concentration"]
        + ["velocity_x", "velocity_y"]
        + [f"D_{i}{j}" for i in range(2) for j in range(2)]
        + [f"E_{i}{j}" for i in range(2) for j in range(2)]
    )
    log(f"{'Channel':<18s} {'Mean':>8s} {'Std':>8s}")
    log(f"{'-'*36}")
    for i, name in enumerate(channel_names):
        flag = "✓" if abs(ch_mean[i]) < 5.0 and ch_std[i] > 0.01 else "✗"
        log(f"{name:<18s} {ch_mean[i]:>+8.3f} {ch_std[i]:>8.3f}  {flag}")

    mean_ok = ch_mean.abs().max() < 5.0
    std_ok = ch_std.min() > 0.01
    log(f"overall mean check : {'✓' if mean_ok else '✗ WARNING: means far from 0'}")
    log(f"overall std check  : {'✓' if std_ok else '✗ WARNING: some channels have near-zero std'}")

    # ── 5. Val/Test no augmentation check ────────────────────────
    print("\n── 5. Val determinism check " + "─" * 29)
    log("Creating val dataset... ", end="")
    t = step_timer()
    ds_val = ActiveMatterDataset(
        args.root, "valid",
        channel_stats=channel_stats,
        label_stats=label_stats,
        return_labels=True,
    )
    print(f"done in {t():.1f}s  ({len(ds_val)} trajectories)")

    log("Loading same sample twice... ", end="")
    f1, l1 = ds_val[0]
    f2, l2 = ds_val[0]
    is_deterministic = torch.equal(f1, f2) and torch.equal(l1, l2)
    print(f"{'✓ identical' if is_deterministic else '✗ FAIL — val should be deterministic'}")

    # ── 6. DataLoader batch ──────────────────────────────────────
    print("\n── 6. DataLoader batch " + "─" * 34)
    log("Creating DataLoader (batch_size=4)... ", end="")
    loader = DataLoader(ds_ssl, batch_size=4, num_workers=0)
    print("done")

    log("Fetching first batch... ", end="")
    t = step_timer()
    x = next(iter(loader))
    print(f"done in {t():.1f}s")

    expected = (4, N_FRAMES, N_CHANNELS, CROP_H, CROP_H)
    assert x.shape == expected, f"FAIL: expected {expected}, got {tuple(x.shape)}"
    log(f"batch shape : {tuple(x.shape)} ✓")
    log(f"value range : min={x.min():.2f}  max={x.max():.2f}  mean={x.mean():.4f}")
    log(f"any NaN     : {'✗ FAIL' if torch.isnan(x).any() else '✓ none'}")
    log(f"any Inf     : {'✗ FAIL' if torch.isinf(x).any() else '✓ none'}")

    # ── 7. build_dataloaders factory ─────────────────────────────
    print("\n── 7. build_dataloaders " + "─" * 33)
    log("Building all three DataLoaders... ", end="")
    t = step_timer()
    train_dl, val_dl, test_dl = build_dataloaders(
        root=args.root,
        batch_size=2,
        num_workers=0,
        channel_stats=channel_stats,
        label_stats=label_stats,
        return_labels=False,
    )
    print(f"done in {t():.1f}s")
    log(f"train : {len(train_dl):>4d} batches  ({len(train_dl.dataset)} samples)")
    log(f"val   : {len(val_dl):>4d} batches  ({len(val_dl.dataset)} samples)")
    log(f"test  : {len(test_dl):>4d} batches  ({len(test_dl.dataset)} samples)")

    # ── 8. Label distribution ────────────────────────────────────
    print("\n── 8. Label distribution (train) " + "─" * 24)
    alphas, zetas = ds_eval.get_all_labels()
    log(f"alpha ({len(set(alphas.tolist()))} unique): {sorted(set(alphas.tolist()))}")
    log(f"zeta  ({len(set(zetas.tolist()))} unique): {sorted(set(zetas.tolist()))}")
    log(f"total combos: {len(set(zip(alphas.tolist(), zetas.tolist())))}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 58}")
    print(f"✓  All dataset pipeline checks passed in {total_elapsed:.1f}s")
    print(f"{'=' * 58}")

    sys.stdout = tee.stdout
    tee.close()
    print(f"\nLog saved to {args.output}")


if __name__ == "__main__":
    main()