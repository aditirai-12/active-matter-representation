#!/usr/bin/env python3
"""Compute CHANNEL_STATS (11 channels) and LABEL_STATS (alpha, zeta) from TRAIN split only.

Usage:
  python scripts/compute_stats.py --root /scratch/$USER/data/active_matter --split train --output data/stats/train_stats.json --crop-size 224
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from typing import Dict, List, Tuple

import h5py
import numpy as np


def find_hdf5_files(root: str, split: str) -> List[str]:
    patterns = [
        os.path.join(root, "data", split, "*.hdf5"),
        os.path.join(root, split, "*.hdf5"),
    ]
    for p in patterns:
        files = sorted(glob.glob(p))
        if files:
            return files
    return []


def center_crop(arr: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = arr.shape[1], arr.shape[2]
    if h == crop_size and w == crop_size:
        return arr
    if h < crop_size or w < crop_size:
        raise ValueError(f"Cannot crop {h}x{w} to {crop_size}x{crop_size}")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return arr[:, top : top + crop_size, left : left + crop_size]


def stack_channels(c: np.ndarray, v: np.ndarray, d: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Return (T, 11, H, W)."""
    c = c[:, np.newaxis, :, :]
    v = v.transpose(0, 3, 1, 2)
    d = d.reshape(d.shape[0], d.shape[1], d.shape[2], 4).transpose(0, 3, 1, 2)
    e = e.reshape(e.shape[0], e.shape[1], e.shape[2], 4).transpose(0, 3, 1, 2)
    return np.concatenate([c, v, d, e], axis=1).astype(np.float64)


def scalar_or_indexed(ds: h5py.Dataset, idx: int) -> float:
    arr = ds[()]
    return float(arr) if np.asarray(arr).ndim == 0 else float(arr[idx])


def compute_stats(files: List[str], crop_size: int | None, time_chunk: int = 8) -> Dict:
    n_channels = 11
    sums = np.zeros(n_channels, dtype=np.float64)
    sq_sums = np.zeros(n_channels, dtype=np.float64)
    counts = np.zeros(n_channels, dtype=np.float64)

    alphas: List[float] = []
    zetas: List[float] = []

    total_files = len(files)
    total_trajs = 0
    total_chunks = 0
    t_start = time.time()

    for file_idx, fpath in enumerate(files):
        file_t = time.time()
        fname = os.path.basename(fpath)

        with h5py.File(fpath, "r") as f:
            conc = f["t0_fields"]["concentration"]  # (N,T,H,W)
            vel = f["t1_fields"]["velocity"]        # (N,T,H,W,2)
            d_t = f["t2_fields"]["D"]               # (N,T,H,W,2,2)
            e_t = f["t2_fields"]["E"]               # (N,T,H,W,2,2)

            n_traj = int(conc.shape[0])
            n_time = int(conc.shape[1])

            print(f"[{file_idx+1}/{total_files}] {fname}  "
                  f"trajs={n_traj}  timesteps={n_time}  "
                  f"shape={list(conc.shape)}")

            for traj_idx in range(n_traj):
                alpha = scalar_or_indexed(f["scalars"]["alpha"], traj_idx)
                zeta = scalar_or_indexed(f["scalars"]["zeta"], traj_idx)
                alphas.append(alpha)
                zetas.append(zeta)

                for t0 in range(0, n_time, time_chunk):
                    t1 = min(t0 + time_chunk, n_time)

                    c = conc[traj_idx, t0:t1]
                    v = vel[traj_idx, t0:t1]
                    d = d_t[traj_idx, t0:t1]
                    e = e_t[traj_idx, t0:t1]

                    if crop_size is not None:
                        c = center_crop(c[..., np.newaxis], crop_size)[..., 0]
                        v = center_crop(v, crop_size)
                        d = center_crop(d, crop_size)
                        e = center_crop(e, crop_size)

                    x = stack_channels(c, v, d, e)
                    sums += x.sum(axis=(0, 2, 3))
                    sq_sums += (x ** 2).sum(axis=(0, 2, 3))
                    counts += x.shape[0] * x.shape[2] * x.shape[3]
                    total_chunks += 1

                total_trajs += 1

        elapsed = time.time() - file_t
        total_elapsed = time.time() - t_start
        files_remaining = total_files - (file_idx + 1)
        avg_per_file = total_elapsed / (file_idx + 1)
        eta = avg_per_file * files_remaining

        print(f"         done in {elapsed:.1f}s  |  "
              f"alpha={alpha:.1f}  zeta={zeta:.1f}  |  "
              f"ETA: {eta/60:.1f} min\n")

    means = sums / counts
    stds = np.sqrt(np.maximum(sq_sums / counts - means ** 2, 1e-12))

    a = np.asarray(alphas, dtype=np.float64)
    z = np.asarray(zetas, dtype=np.float64)

    total_time = time.time() - t_start

    print("=" * 60)
    print(f"DONE  |  {total_files} files  |  {total_trajs} trajectories  |  "
          f"{total_chunks} chunks  |  {total_time:.1f}s total")
    print("=" * 60)

    # Print per-channel stats as a table
    channel_names = (
        ["concentration"]
        + ["velocity_x", "velocity_y"]
        + [f"D_{i}{j}" for i in range(2) for j in range(2)]
        + [f"E_{i}{j}" for i in range(2) for j in range(2)]
    )
    print(f"\n{'Channel':<20s} {'Mean':>12s} {'Std':>12s}")
    print("-" * 46)
    for i, name in enumerate(channel_names):
        print(f"  {name:<18s} {means[i]:>+12.6f} {stds[i]:>12.6f}")

    print(f"\nLabels:")
    print(f"  alpha  unique={sorted(set(alphas))}  mean={a.mean():.4f}  std={a.std():.4f}")
    print(f"  zeta   unique={sorted(set(zetas))}   mean={z.mean():.4f}  std={z.std():.4f}")

    return {
        "channel": {
            "mean": means.tolist(),
            "std": stds.tolist(),
        },
        "label": {
            "alpha": {"mean": float(a.mean()), "std": float(max(a.std(), 1e-12))},
            "zeta": {"mean": float(z.mean()), "std": float(max(z.std(), 1e-12))},
        },
        "meta": {
            "num_files": len(files),
            "split": "train",
            "crop_size": crop_size,
            "num_labels": int(len(alphas)),
            "total_trajectories": total_trajs,
            "compute_time_seconds": round(total_time, 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root (contains train split)")
    parser.add_argument("--split", type=str, default="train", choices=["train"], help="Must be train")
    parser.add_argument("--output", type=str, default="data/stats/train_stats.json")
    parser.add_argument("--crop-size", type=int, default=224, help="Use 224 to match model input, or 0 for no crop")
    parser.add_argument("--time-chunk", type=int, default=8)
    args = parser.parse_args()

    print(f"Root:       {args.root}")
    print(f"Split:      {args.split}")
    print(f"Crop size:  {args.crop_size}")
    print(f"Output:     {args.output}")
    print()

    files = find_hdf5_files(args.root, args.split)
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found for split '{args.split}' under {args.root}")

    print(f"Found {len(files)} HDF5 files\n")

    crop_size = None if args.crop_size == 0 else args.crop_size
    stats = compute_stats(files, crop_size=crop_size, time_chunk=args.time_chunk)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved stats -> {args.output}")
    print("\nPaste into src/dataset.py:")
    print("CHANNEL_STATS =", json.dumps(stats["channel"], indent=2))
    print("LABEL_STATS   =", json.dumps(stats["label"], indent=2))


if __name__ == "__main__":
    main()