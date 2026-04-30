#!/usr/bin/env python3
"""Check for representation collapse across saved encoder checkpoints.

Usage:
  python scripts/collapse_check.py --encoder_path checkpoints/.../ConvEncoder_3.pth --root /scratch/$USER/data/active_matter --stats data/stats/train_stats.json

Checks embedding std across val batches. Healthy values > 0.1.
"""

import argparse
import sys
import os

import torch
import numpy as np

sys.path.insert(0, ".")

from src.dataset import ActiveMatterDataset, load_stats
from physics_jepa.data import ActiveMatterJEPADataset
from physics_jepa.model import get_model_and_loss_cnn
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, required=True, help="Path to saved ConvEncoder .pth file")
    parser.add_argument("--root", type=str, default="/scratch/" + os.environ.get("USER", "ar9598") + "/data/active_matter")
    parser.add_argument("--stats", type=str, default="data/stats/train_stats.json")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of val batches to check")
    parser.add_argument("--batch_size", type=int, default=4)
    # model config — should match whatever was used for training
    parser.add_argument("--dims", type=int, nargs="+", default=[16, 32, 64, 128, 128])
    parser.add_argument("--num_res_blocks", type=int, nargs="+", default=[3, 3, 3, 9, 3])
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--in_chans", type=int, default=11)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Encoder: {args.encoder_path}\n")

    # Load stats
    channel_stats, label_stats = load_stats(args.stats)

    # Build val dataset
    base_ds = ActiveMatterDataset(
        root=args.root,
        split="valid",
<<<<<<< HEAD
        n_frames=2 * (args.num_frames // 2),  # need context + target
=======
        n_frames=2 * args.num_frames,  # need context + target
>>>>>>> 2613d82 (adding eval scripts and results)
        return_labels=False,
        channel_stats=channel_stats,
        label_stats=label_stats,
        augment=False,
        random_temporal_crop=False,
    )
<<<<<<< HEAD
    jepa_ds = ActiveMatterJEPADataset(base_ds, include_labels=False, num_frames=args.num_frames // 2)
=======
    jepa_ds = ActiveMatterJEPADataset(base_ds, include_labels=False, num_frames=args.num_frames)
>>>>>>> 2613d82 (adding eval scripts and results)
    loader = DataLoader(jepa_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Build encoder with same config as training
    encoder, _, _ = get_model_and_loss_cnn(
        dims=args.dims,
        num_res_blocks=args.num_res_blocks,
<<<<<<< HEAD
        num_frames=args.num_frames // 2,
=======
        num_frames=args.num_frames,
>>>>>>> 2613d82 (adding eval scripts and results)
        in_chans=args.in_chans,
    )

    # Load weights
    state_dict = torch.load(args.encoder_path, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
<<<<<<< HEAD
    encoder.load_state_dict(state_dict)
=======
    encoder.load_state_dict(state_dict, strict=False)
>>>>>>> 2613d82 (adding eval scripts and results)
    encoder.to(device)
    encoder.eval()

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Checking {args.num_batches} batches...\n")

    all_stds = []
    all_means = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break

            ctx = batch["context"].to(device)
            embed = encoder(ctx)

            # std and mean across batch dim, per feature
            batch_std = embed.float().std(dim=0).mean().item()
            batch_mean = embed.float().mean().item()
            all_stds.append(batch_std)
            all_means.append(batch_mean)

            print(f"  batch {i}: embed shape={tuple(embed.shape)}  std={batch_std:.4f}  mean={batch_mean:.4f}")

    avg_std = np.mean(all_stds)
    avg_mean = np.mean(all_means)

    print(f"\n{'=' * 50}")
    print(f"Average embedding std:  {avg_std:.4f}")
    print(f"Average embedding mean: {avg_mean:.4f}")

    if avg_std > 0.1:
        print(f"No representation collapse detected (std={avg_std:.4f} > 0.1)")
    elif avg_std > 0.01:
        print(f"WARNING: Low embedding std ({avg_std:.4f}) — possible partial collapse")
    else:
        print(f"COLLAPSE DETECTED: embedding std={avg_std:.4f} ≈ 0")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
