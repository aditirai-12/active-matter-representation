"""
dataset.py
==========
Phase 2 data pipeline for active_matter representation learning.

Covers
------
- Dataset class  (loads HDF5 files from HuggingFace download or mock)
- Per-channel z-score normalisation for all 11 physical channels
- Spatial & temporal augmentations (random crop, flip, temporal jitter)
- DataLoader factory with train / val / test splits
- Label leakage prevention: labels are NEVER returned during SSL pre-training

HDF5 file structure (real data + mock from create_mock_data.py)
---------------------------------------------------------------
    scalars/
        alpha               float32 — active dipole strength   (label, withheld)
        zeta                float32 — steric alignment         (label, withheld)
    t0_fields/
        concentration       (N_TRAJ, T, H, W)          →  1 channel
    t1_fields/
        velocity            (N_TRAJ, T, H, W, 2)        →  2 channels
    t2_fields/
        D                   (N_TRAJ, T, H, W, 2, 2)     →  4 channels  (strain-rate)
        E                   (N_TRAJ, T, H, W, 2, 2)     →  4 channels  (orientation)

Output tensor shape:  (T=16, C=11, H=224, W=224)  float32
"""

import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_FRAMES:   int = 16   # temporal window per sample (project spec)
RAW_T:      int = 81   # time-steps per raw trajectory
RAW_H:      int = 256  # raw spatial size
CROP_H:     int = 224  # target spatial size (project spec)
N_CHANNELS: int = 11   # total physical channels

# Channel layout (concatenation order):
#   [0]      concentration
#   [1–2]    velocity x, y
#   [3–6]    strain-rate D (2×2 flattened)
#   [7–10]   orientation E (2×2 flattened)
CHANNEL_NAMES: List[str] = (
    ["concentration"]
    + ["velocity_x", "velocity_y"]
    + [f"D_{i}{j}" for i in range(2) for j in range(2)]
    + [f"E_{i}{j}" for i in range(2) for j in range(2)]
)

# Per-channel z-score stats computed from training data (data/stats/train_stats.json).
# Channels: concentration, velocity_x, velocity_y, D_00, D_01, D_10, D_11, E_00, E_01, E_10, E_11
CHANNEL_STATS: Dict[str, List[float]] = {
    "mean": [
        1.0000147,   #  concentration
        0.0073079,   #  velocity_x
       -0.0057337,   #  velocity_y
        0.5024746,   #  D_00
       -0.0076769,   #  D_01
       -0.0076769,   #  D_10
        0.4975401,   #  D_11
        0.0002608,   #  E_00
       -0.0011301,   #  E_01
       -0.0011301,   #  E_10
       -0.0002608,   #  E_11
    ],
    "std": [
        0.0028454,   #  concentration  ← very tight; normalization matters a lot here
        0.5740187,   #  velocity_x
        0.5643068,   #  velocity_y
        0.3203980,   #  D_00
        0.3308609,   #  D_01
        0.3308609,   #  D_10
        0.3204363,   #  D_11
        0.4073318,   #  E_00
        0.4315492,   #  E_01
        0.4315492,   #  E_10
        0.4073318,   #  E_11
    ],
}

# Label z-score stats computed from training data (data/stats/train_stats.json).
LABEL_STATS: Dict[str, Dict[str, float]] = {
    "alpha": {"mean": -3.0228571, "std": 1.4538787},
    "zeta":  {"mean":  9.0228571, "std": 5.2109273},
}


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────

def _find_files(root: str, split: str) -> List[str]:
    """
    Locate HDF5 files. Tries multiple layouts so both real and mock data work:
      1. <root>/data/<split>/*.hdf5   — real HuggingFace download
      2. <root>/<split>/*.hdf5
      3. <root>/data/mock/*.hdf5      — create_mock_data.py output
      4. <root>/mock/*.hdf5
    """
    patterns = [
        os.path.join(root, "data", split,   "*.hdf5"),
        os.path.join(root, split,            "*.hdf5"),
        os.path.join(root, "data", "mock",  "*.hdf5"),
        os.path.join(root, "mock",           "*.hdf5"),
    ]
    for p in patterns:
        files = sorted(glob.glob(p))
        if files:
            return files
    return []


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 reading
# ─────────────────────────────────────────────────────────────────────────────

def _n_trajectories(f: h5py.File) -> int:
    return int(f["t0_fields"]["concentration"].shape[0])


def _n_timesteps(f: h5py.File) -> int:
    return int(f["t0_fields"]["concentration"].shape[1])


def _read_param(scalars: h5py.Group, name: str, traj_idx: int) -> float:
    """Read alpha or zeta — handles scalar () and 1-D (N,) datasets."""
    arr = scalars[name][()]
    return float(arr) if arr.ndim == 0 else float(arr[traj_idx])


def _load_raw_window(
    f: h5py.File,
    traj_idx: int,
    t_start: int,
    n_frames: int,
) -> np.ndarray:
    """
    Load one temporal window, concatenating all 11 physical channels.
    Returns float32 array of shape (n_frames, 11, H, W)  — un-cropped.
    """
    i = traj_idx
    t = slice(t_start, t_start + n_frames)

    # concentration: (T, H, W) → (T, 1, H, W)
    c = f["t0_fields"]["concentration"][i, t]
    c = c[:, np.newaxis, :, :]

    # velocity: (T, H, W, 2) → (T, 2, H, W)
    v = f["t1_fields"]["velocity"][i, t]
    v = v.transpose(0, 3, 1, 2)

    # D (strain-rate): (T, H, W, 2, 2) → (T, 4, H, W)
    D = f["t2_fields"]["D"][i, t]
    D = D.reshape(D.shape[0], D.shape[1], D.shape[2], 4).transpose(0, 3, 1, 2)

    # E (orientation): (T, H, W, 2, 2) → (T, 4, H, W)
    E = f["t2_fields"]["E"][i, t]
    E = E.reshape(E.shape[0], E.shape[1], E.shape[2], 4).transpose(0, 3, 1, 2)

    return np.concatenate([c, v, D, E], axis=1).astype(np.float32)  # (T,11,H,W)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics computation (run once on training data, then hard-code results)
# ─────────────────────────────────────────────────────────────────────────────

def compute_channel_stats_fast(
    root: str, split: str = "train", max_files: int = 10
) -> Dict:
    """
    Compute per-channel mean and std across training trajectories.

    Run this ONCE on the real training data, then paste the printed values
    into CHANNEL_STATS at the top of this file.

    Parameters
    ----------
    root      : dataset root
    split     : always use "train"
    max_files : how many files to scan (use all for final numbers, subset for speed)
    """
    files = _find_files(root, split)[:max_files]
    if not files:
        raise FileNotFoundError(f"No files for split='{split}' under {root}")

    sums    = np.zeros(N_CHANNELS, np.float64)
    sq_sums = np.zeros(N_CHANNELS, np.float64)
    counts  = np.zeros(N_CHANNELS, np.float64)

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            n_traj = _n_trajectories(f)
            n_time = _n_timesteps(f)
            for ti in range(n_traj):
                frames = _load_raw_window(f, ti, 0, min(n_time, N_FRAMES)).astype(np.float64)
                sums    += frames.sum(axis=(0, 2, 3))
                sq_sums += (frames ** 2).sum(axis=(0, 2, 3))
                counts  += frames.shape[0] * frames.shape[2] * frames.shape[3]

    mean = sums / counts
    std  = np.sqrt(np.maximum(sq_sums / counts - mean ** 2, 1e-8))
    stats = {"mean": mean.tolist(), "std": std.tolist()}

    print("Channel stats — paste into CHANNEL_STATS:")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"  {name:20s}  mean={stats['mean'][i]:+.4f}  std={stats['std'][i]:.4f}")
    return stats


def compute_label_stats(root: str, split: str = "train") -> Dict:
    """
    Compute mean/std for alpha and zeta labels from all training trajectories.
    Run once, paste results into LABEL_STATS.
    """
    files = _find_files(root, split)
    if not files:
        raise FileNotFoundError(f"No files for split='{split}' under {root}")

    alphas, zetas = [], []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            n = _n_trajectories(f)
            for i in range(n):
                alphas.append(_read_param(f["scalars"], "alpha", i))
                zetas.append(_read_param(f["scalars"], "zeta",  i))

    a = np.array(alphas, np.float32)
    z = np.array(zetas,  np.float32)
    stats = {
        "alpha": {"mean": float(a.mean()), "std": max(float(a.std()), 1e-6)},
        "zeta":  {"mean": float(z.mean()), "std": max(float(z.std()), 1e-6)},
    }
    print(f"Label stats — paste into LABEL_STATS:")
    print(f"  alpha: mean={stats['alpha']['mean']:.4f}  std={stats['alpha']['std']:.4f}")
    print(f"  zeta:  mean={stats['zeta']['mean']:.4f}   std={stats['zeta']['std']:.4f}")
    return stats


def save_stats(channel_stats: Dict, label_stats: Dict, path: str) -> None:
    """Save computed stats to JSON for reproducibility."""
    with open(path, "w") as fp:
        json.dump({"channel": channel_stats, "label": label_stats}, fp, indent=2)
    print(f"Stats saved → {path}")


def load_stats(path: str) -> Tuple[Dict, Dict]:
    """Load previously saved stats JSON."""
    with open(path) as fp:
        d = json.load(fp)
    return d["channel"], d["label"]


# ─────────────────────────────────────────────────────────────────────────────
# Augmentations
# ─────────────────────────────────────────────────────────────────────────────

def _center_crop(frames: np.ndarray) -> np.ndarray:
    """(T, C, H, W) — deterministic centre crop 256→224."""
    off = (RAW_H - CROP_H) // 2
    return frames[:, :, off:off + CROP_H, off:off + CROP_H]


def _random_spatial_crop(frames: np.ndarray) -> np.ndarray:
    """(T, C, H, W) — random spatial crop 256→224."""
    h, w = frames.shape[2], frames.shape[3]
    if h == CROP_H:
        return frames
    top  = np.random.randint(0, h - CROP_H + 1)
    left = np.random.randint(0, w - CROP_H + 1)
    return frames[:, :, top:top + CROP_H, left:left + CROP_H]


# Channels whose sign flips under x -> -x (horizontal): v_x and tensor off-diagonals
_HFLIP_SIGN_FLIP_CHANNELS = np.array([1, 4, 5, 8, 9])
# Channels whose sign flips under y -> -y (vertical): v_y and tensor off-diagonals
_VFLIP_SIGN_FLIP_CHANNELS = np.array([2, 4, 5, 8, 9])


def _random_flip(frames: np.ndarray) -> np.ndarray:
    """Random horizontal/vertical flip with sign correction for vector/tensor channels.

    Frames are (T, C, H, W) with the active_matter channel layout:
      [0] concentration, [1,2] velocity, [3-6] D tensor, [7-10] E tensor.
    A spatial reflection x -> -x must negate v_x and the tensor off-diagonals
    (D_xy, D_yx, E_xy, E_yx); analogously for y -> -y.
    """
    if np.random.rand() < 0.5:
        frames = frames[:, :, :, ::-1].copy()    # horizontal: x -> -x
        frames[:, _HFLIP_SIGN_FLIP_CHANNELS] *= -1
    if np.random.rand() < 0.5:
        frames = frames[:, :, ::-1, :].copy()    # vertical: y -> -y
        frames[:, _VFLIP_SIGN_FLIP_CHANNELS] *= -1
    return frames


def _apply_augmentations(frames: np.ndarray, training: bool) -> np.ndarray:
    """
    Full spatial augmentation pipeline.
    Training : random crop (256→224) + random H/V flips
    Val/Test : deterministic centre crop, no flips
    """
    if training:
        frames = _random_spatial_crop(frames)
        frames = _random_flip(frames)
    else:
        if frames.shape[2] == RAW_H:
            frames = _center_crop(frames)
    return np.ascontiguousarray(frames)


def _normalise_channels(frames: np.ndarray, stats: Dict) -> np.ndarray:
    """Z-score normalise each of the 11 channels independently."""
    mean = np.array(stats["mean"], np.float32)[np.newaxis, :, np.newaxis, np.newaxis]
    std  = np.array(stats["std"],  np.float32)[np.newaxis, :, np.newaxis, np.newaxis]
    return (frames - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ActiveMatterDataset(Dataset):
    """
    PyTorch Dataset for the active_matter subset of The Well.

    ┌─────────────────────────────────────────────────────────────────┐
    │  LABEL LEAKAGE PREVENTION                                       │
    │  return_labels=False (default) during SSL pre-training.         │
    │  Labels (alpha, zeta) are NEVER used to update model weights.   │
    │  Only set return_labels=True for linear probe / kNN eval.       │
    └─────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    root : str
        Dataset root (contains data/train, data/valid, data/test  OR  data/mock).
    split : "train" | "valid" | "test"
        Which split to load.
    n_frames : int
        Temporal window size — default 16 (project spec).
    return_labels : bool
        False → returns tensor only  ← SSL pre-training (no label leakage)
        True  → returns (tensor, labels)  ← evaluation only
    normalize_channels : bool
        Apply per-channel z-score normalisation to all 11 channels.
    channel_stats : dict | None
        {"mean": [11 floats], "std": [11 floats]}.
        If None, uses placeholder (all zeros / ones).
    normalize_labels : bool
        Z-score normalise alpha / zeta (required by project spec).
    label_stats : dict | None
        {"alpha": {"mean":…,"std":…}, "zeta":{…}}.
        If None, uses placeholder.
    augment : bool | None
        True → training augmentations. None → auto (True if split=="train").
    random_temporal_crop : bool | None
        True → random start frame.  None → auto (True if split=="train").

    Returns from __getitem__
    ------------------------
    frames : FloatTensor  (T=16, C=11, H=224, W=224)
    labels : FloatTensor  (2,)  = [alpha_zscore, zeta_zscore]
             Only when return_labels=True.
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "valid", "test"] = "train",
        n_frames: int = N_FRAMES,
        return_labels: bool = False,
        normalize_channels: bool = True,
        channel_stats: Optional[Dict] = None,
        normalize_labels: bool = True,
        label_stats: Optional[Dict] = None,
        augment: Optional[bool] = None,
        random_temporal_crop: Optional[bool] = None,
    ):
        super().__init__()
        self.root   = root
        self.split  = split
        self.n_frames = n_frames

        self.return_labels      = return_labels
        self.normalize_channels = normalize_channels
        self.channel_stats      = channel_stats or CHANNEL_STATS
        self.normalize_labels   = normalize_labels
        self.label_stats        = label_stats or LABEL_STATS

        is_train = (split == "train")
        self.augment              = augment              if augment              is not None else is_train
        self.random_temporal_crop = random_temporal_crop if random_temporal_crop is not None else is_train

        self.files = _find_files(root, split)
        if not self.files:
            raise FileNotFoundError(
                f"No .hdf5 files for split='{split}' under '{root}'.\n"
                f"Mock:  python scripts/create_mock_data.py\n"
                f"Real:  hf download polymathic-ai/active_matter "
                f"--repo-type dataset --local-dir {root}"
            )

        self._index: List[Tuple[str, int, float, float]] = []
        self._build_index()

    def _build_index(self) -> None:
        for fpath in self.files:
            with h5py.File(fpath, "r") as f:
                n = _n_trajectories(f)
                for i in range(n):
                    alpha = _read_param(f["scalars"], "alpha", i)
                    zeta  = _read_param(f["scalars"], "zeta",  i)
                    self._index.append((fpath, i, alpha, zeta))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        fpath, traj_idx, alpha_raw, zeta_raw = self._index[idx]

        # 1. load raw frames
        with h5py.File(fpath, "r") as f:
            n_time  = _n_timesteps(f)
            t_start = self._pick_t_start(n_time)
            frames  = _load_raw_window(f, traj_idx, t_start, self.n_frames)

        # 2. spatial augmentation (crop + optional flips)
        frames = _apply_augmentations(frames, training=self.augment)

        # 3. per-channel z-score normalisation
        if self.normalize_channels:
            frames = _normalise_channels(frames, self.channel_stats)

        frames = torch.from_numpy(frames)   # (T, 11, H, W)

        # 4. labels — only for eval, never during SSL training
        if self.return_labels:
            return frames, self._make_labels(alpha_raw, zeta_raw)
        return frames

    def _pick_t_start(self, n_time: int) -> int:
        max_start = max(n_time - self.n_frames, 0)
        if self.random_temporal_crop and max_start > 0:
            return int(torch.randint(0, max_start + 1, (1,)).item())
        return 0

    def _make_labels(self, alpha: float, zeta: float) -> torch.Tensor:
        if self.normalize_labels:
            alpha = (alpha - self.label_stats["alpha"]["mean"]) / self.label_stats["alpha"]["std"]
            zeta  = (zeta  - self.label_stats["zeta"]["mean"])  / self.label_stats["zeta"]["std"]
        return torch.tensor([alpha, zeta], dtype=torch.float32)

    def get_all_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw (alpha, zeta) arrays — useful for checking label distribution."""
        alphas = np.array([e[2] for e in self._index], np.float32)
        zetas  = np.array([e[3] for e in self._index], np.float32)
        return alphas, zetas

    def __repr__(self) -> str:
        return (
            f"ActiveMatterDataset(split={self.split!r}, n={len(self)}, "
            f"n_frames={self.n_frames}, return_labels={self.return_labels}, "
            f"augment={self.augment}, norm_ch={self.normalize_channels})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    n_frames: int = N_FRAMES,
    channel_stats: Optional[Dict] = None,
    label_stats: Optional[Dict] = None,
    return_labels: bool = False,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    SSL pre-training   → return_labels=False (default, no label leakage)
    Linear probe/kNN   → return_labels=True

    Train: random temporal crop + random spatial crop + random flips + shuffle.
    Val / Test: deterministic centre crop, no flips, no shuffle.
    """
    common = dict(
        n_frames=n_frames,
        channel_stats=channel_stats,
        label_stats=label_stats,
        return_labels=return_labels,
        normalize_channels=True,
    )
    train_ds = ActiveMatterDataset(root, split="train", **common)
    val_ds   = ActiveMatterDataset(root, split="valid", **common)
    test_ds  = ActiveMatterDataset(root, split="test",  **common)

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    return (
        DataLoader(train_ds, shuffle=True,  drop_last=True,  **kw),
        DataLoader(val_ds,   shuffle=False, drop_last=False, **kw),
        DataLoader(test_ds,  shuffle=False, drop_last=False, **kw),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test — python src/dataset.py <root>
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, time

    root  = sys.argv[1] if len(sys.argv) > 1 else "."
    split = "mock" if (_find_files(root, "mock") and not _find_files(root, "train")) else "train"
    print(f"Root : {root}")
    print(f"Split: {split}\n")

    print("── 1. label stats " + "─" * 37)
    lstats = compute_label_stats(root, split)

    print("\n── 2. channel stats " + "─" * 35)
    cstats = compute_channel_stats_fast(root, split, max_files=5)

    print("\n── 3. dataset sizes " + "─" * 35)
    for sp in ("train", "valid", "test"):
        try:
            ds = ActiveMatterDataset(root, sp, channel_stats=cstats, label_stats=lstats)
            print(f"  {sp:5s}: {len(ds)} trajectories")
        except FileNotFoundError:
            ds = ActiveMatterDataset(root, "mock", channel_stats=cstats, label_stats=lstats)
            print(f"  mock : {len(ds)} trajectories (real splits not downloaded yet)")
            break

    print("\n── 4. SSL mode — no labels " + "─" * 27)
    ds_ssl = ActiveMatterDataset(
        root, split, channel_stats=cstats, label_stats=lstats, return_labels=False
    )
    t0   = time.perf_counter()
    item = ds_ssl[0]
    ms   = (time.perf_counter() - t0) * 1000
    assert not isinstance(item, tuple), "FAIL: labels returned in SSL mode!"
    print(f"  return type : {type(item).__name__} (not a tuple ✓  — no label leakage)")
    print(f"  shape       : {tuple(item.shape)}")
    print(f"  load time   : {ms:.1f} ms")
    assert tuple(item.shape) == (N_FRAMES, N_CHANNELS, CROP_H, CROP_H)

    print("\n── 5. eval mode — with labels " + "─" * 24)
    ds_eval = ActiveMatterDataset(
        root, split, channel_stats=cstats, label_stats=lstats, return_labels=True
    )
    frames, labels = ds_eval[0]
    print(f"  frames : {tuple(frames.shape)}")
    print(f"  labels : {labels}  ← [alpha_z, zeta_z]")
    assert tuple(labels.shape) == (2,)

    print("\n── 6. channel normalisation " + "─" * 26)
    batch = torch.stack([ds_eval[i][0] for i in range(min(8, len(ds_eval)))])
    ch_mean = batch.mean(dim=(0, 1, 3, 4))
    ch_std  = batch.std(dim=(0, 1, 3, 4))
    print(f"  per-channel mean range : [{ch_mean.min():.3f}, {ch_mean.max():.3f}]  (expect ~0)")
    print(f"  per-channel std  range : [{ch_std.min():.3f},  {ch_std.max():.3f}]   (expect ~1)")

    print("\n── 7. DataLoader batch " + "─" * 31)
    loader = DataLoader(ds_ssl, batch_size=4, num_workers=0)
    x = next(iter(loader))
    assert x.shape == (4, N_FRAMES, N_CHANNELS, CROP_H, CROP_H)
    print(f"  batch shape : {tuple(x.shape)} ✓")

    print("\n── 8. alpha/zeta distribution " + "─" * 23)
    alphas, zetas = ds_eval.get_all_labels()
    print(f"  alpha unique values : {sorted(set(alphas.tolist()))}")
    print(f"  zeta  unique values : {sorted(set(zetas.tolist()))}")

    print("\n" + "═" * 55)
    print("✓  All Phase 2 checks passed.")
    print("═" * 55)
    print()
    print("TODO: run on real training data and paste the printed")
    print("channel + label stats into CHANNEL_STATS / LABEL_STATS")
    print("at the top of this file (or save to stats.json and load")
    print("with load_stats()).")