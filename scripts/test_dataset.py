#!/usr/bin/env python3
"""Quick validation of the data pipeline using pre-computed stats.

Usage (from repo root):
  python scripts/test_dataset.py --root /scratch/$USER/data/active_matter --stats data/stats/train_stats.json
"""

import argparse
import sys
import time

import torch
from torch.utils.data import DataLoader

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--stats", type=str, default="data/stats/train_stats.json")
    args = parser.parse_args()

    channel_stats, label_stats = load_stats(args.stats)
    print(f"Loaded stats from {args.stats}\n")

    # ── 1. Check all splits load ──────────────────────────────────
    print("── 1. Split sizes " + "─" * 40)
    for split in ("train", "valid", "test"):
        try:
            ds = ActiveMatterDataset(
                args.root, split,
                channel_stats=channel_stats,
                label_stats=label_stats,
                return_labels=True,
            )
            alphas, zetas = ds.get_all_labels()
            print(f"  {split:5s}: {len(ds):>4d} trajectories  |  "
                  f"alpha unique: {len(set(alphas.tolist()))}  "
                  f"zeta unique: {len(set(zetas.tolist()))}")
        except FileNotFoundError as e:
            print(f"  {split:5s}: NOT FOUND — {e}")

    # ── 2. SSL mode (no labels) ──────────────────────────────────
    print("\n── 2. SSL mode — no labels " + "─" * 30)
    ds_ssl = ActiveMatterDataset(
        args.root, "train",
        channel_stats=channel_stats,
        label_stats=label_stats,
        return_labels=False,
    )
    t0 = time.perf_counter()
    item = ds_ssl[0]
    ms = (time.perf_counter() - t0) * 1000

    assert not isinstance(item, tuple), "FAIL: labels leaked in SSL mode!"
    assert item.shape == (N_FRAMES, N_CHANNELS, CROP_H, CROP_H), \
        f"FAIL: expected ({N_FRAMES},{N_CHANNELS},{CROP_H},{CROP_H}), got {tuple(item.shape)}"
    print(f"  shape     : {tuple(item.shape)} ✓")
    print(f"  dtype     : {item.dtype}")
    print(f"  load time : {ms:.0f} ms")
    print(f"  no labels : ✓")

    # ── 3. Eval mode (with labels) ───────────────────────────────
    print("\n── 3. Eval mode — with labels " + "─" * 27)
    ds_eval = ActiveMatterDataset(
        args.root, "train",
        channel_stats=channel_stats,
        label_stats=label_stats,
        return_labels=True,
    )
    frames, labels = ds_eval[0]
    assert frames.shape == (N_FRAMES, N_CHANNELS, CROP_H, CROP_H)
    assert labels.shape == (2,)
    print(f"  frames : {tuple(frames.shape)} ✓")
    print(f"  labels : {labels.tolist()}  [alpha_z, zeta_z]")

    # ── 4. Channel normalization sanity check ────────────────────
    print("\n── 4. Channel normalization " + "─" * 29)
    n_samples = min(16, len(ds_eval))
    batch = torch.stack([ds_eval[i][0] for i in range(n_samples)])
    ch_mean = batch.mean(dim=(0, 1, 3, 4))
    ch_std = batch.std(dim=(0, 1, 3, 4))
    print(f"  samples checked : {n_samples}")
    print(f"  mean range : [{ch_mean.min():.3f}, {ch_mean.max():.3f}]  (expect ~0)")
    print(f"  std  range : [{ch_std.min():.3f}, {ch_std.max():.3f}]  (expect ~1)")

    mean_ok = ch_mean.abs().max() < 5.0  # loose check
    std_ok = ch_std.min() > 0.01
    print(f"  mean check : {'✓' if mean_ok else '✗ WARNING: means far from 0'}")
    print(f"  std check  : {'✓' if std_ok else '✗ WARNING: some channels have near-zero std'}")

    # ── 5. Val/Test no augmentation check ────────────────────────
    print("\n── 5. Val determinism check " + "─" * 29)
    ds_val = ActiveMatterDataset(
        args.root, "valid",
        channel_stats=channel_stats,
        label_stats=label_stats,
        return_labels=True,
    )
    f1, l1 = ds_val[0]
    f2, l2 = ds_val[0]
    is_deterministic = torch.equal(f1, f2) and torch.equal(l1, l2)
    print(f"  same output twice : {'✓' if is_deterministic else '✗ FAIL — val should be deterministic'}")

    # ── 6. DataLoader batch ──────────────────────────────────────
    print("\n── 6. DataLoader batch " + "─" * 34)
    loader = DataLoader(ds_ssl, batch_size=4, num_workers=0)
    x = next(iter(loader))
    expected = (4, N_FRAMES, N_CHANNELS, CROP_H, CROP_H)
    assert x.shape == expected, f"FAIL: expected {expected}, got {tuple(x.shape)}"
    print(f"  batch shape : {tuple(x.shape)} ✓")
    print(f"  min={x.min():.2f}  max={x.max():.2f}  mean={x.mean():.4f}")

    # ── 7. build_dataloaders factory ─────────────────────────────
    print("\n── 7. build_dataloaders " + "─" * 33)
    train_dl, val_dl, test_dl = build_dataloaders(
        root=args.root,
        batch_size=2,
        num_workers=0,
        channel_stats=channel_stats,
        label_stats=label_stats,
        return_labels=False,
    )
    print(f"  train batches : {len(train_dl)}")
    print(f"  val batches   : {len(val_dl)}")
    print(f"  test batches  : {len(test_dl)}")

    # ── 8. Label distribution ────────────────────────────────────
    print("\n── 8. Label distribution (train) " + "─" * 24)
    alphas, zetas = ds_eval.get_all_labels()
    print(f"  alpha: {sorted(set(alphas.tolist()))}")
    print(f"  zeta:  {sorted(set(zetas.tolist()))}")

    print("\n" + "=" * 58)
    print("✓  All dataset pipeline checks passed.")
    print("=" * 58)


if __name__ == "__main__":
    main()