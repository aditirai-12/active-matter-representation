import argparse
import copy
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf

from .train import Trainer
from .utils.hydra import compose
from .utils.misc import distprint


class JepaTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.target_encoder = None   # created in on_training_start
        self._ema_schedule = None
        self._ema_step = 0

    # ---- Lifecycle hooks ------------------------------------------------

    def on_training_start(self, model_components, total_steps):
        """Create the EMA target encoder and momentum schedule."""
        encoder = model_components[0]

        # Deep-copy encoder weights; no gradients needed
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.to(self.rank)

        # Cosine schedule for momentum: ema_start -> ema_end over training
        ema_start = self.train_cfg.get("ema_start", 0.996)
        ema_end   = self.train_cfg.get("ema_end",   1.0)
        grad_accum_steps = self.set_up_gradient_accumulation()
        total_updates = (total_steps + grad_accum_steps - 1) // grad_accum_steps
        # Cosine interpolation from ema_start to ema_end
        t = np.arange(total_updates, dtype=np.float32) / max(total_updates - 1, 1)
        self._ema_schedule = ema_end - (ema_end - ema_start) * (1 + np.cos(np.pi * t)) / 2
        self._ema_step = 0

        distprint(
            f"EMA target encoder: tau {ema_start:.4f} -> {ema_end:.4f} over {total_updates} updates",
            local_rank=self.rank,
        )

        # If resuming, try to load target encoder checkpoint
        if 'target_encoder_path' in self.train_cfg and self.train_cfg.target_encoder_path is not None:
            distprint(f"loading target encoder from {self.train_cfg.target_encoder_path}", local_rank=self.rank)
            state_dict = torch.load(self.train_cfg.target_encoder_path, map_location=f"cuda:{self.rank}")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.target_encoder.load_state_dict(state_dict)

    @torch.no_grad()
    def on_after_optimizer_step(self):
        """Update target encoder via exponential moving average."""
        if self.target_encoder is None:
            return

        tau = float(self._ema_schedule[min(self._ema_step, len(self._ema_schedule) - 1)])
        self._ema_step += 1

        for p_target, p_online in zip(
            self.target_encoder.parameters(),
            self._get_encoder().parameters(),
        ):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1 - tau)

    def save_extra_state(self, out_path, tag):
        """Save the EMA target encoder alongside the regular checkpoint."""
        if self.target_encoder is not None:
            torch.save(
                self.target_encoder.state_dict(),
                out_path / f"TargetEncoder_{tag}.pth",
            )

    # ---- Helpers --------------------------------------------------------

    def _get_encoder(self):
        """Return the online encoder, unwrapping DDP if needed."""
        enc = self._online_encoder_ref
        return enc.module if hasattr(enc, 'module') else enc

    # ---- Core JEPA forward pass -----------------------------------------

    def pred_fn(self, batch, model_components, loss_fn):
        encoder, predictor = model_components

        # Stash reference so on_after_optimizer_step can find the encoder
        self._online_encoder_ref = encoder

        # Online encoder processes context (gradients flow here)
        ctx_embed = encoder(batch['context'])

        # EMA target encoder processes target (no gradients)
        if self.target_encoder is not None:
            with torch.no_grad():
                tgt_embed = self.target_encoder(batch['target'])
        else:
            # Fallback before on_training_start runs
            tgt_embed = encoder(batch['target'])

        pred = predictor(ctx_embed)

        # Compute loss on predicted vs target embeddings
        if len(pred.shape) < 5:
            loss_dict = loss_fn(pred.unsqueeze(2), tgt_embed.unsqueeze(2))
        else:
            loss_dict = loss_fn(pred, tgt_embed)

        # Representation collapse check
        with torch.no_grad():
            ctx_std = ctx_embed.float().std(dim=0).mean()
            tgt_std = tgt_embed.float().std(dim=0).mean()
            loss_dict['ctx_embed_std'] = ctx_std
            loss_dict['tgt_embed_std'] = tgt_std

            # Log current EMA momentum
            if self._ema_schedule is not None:
                idx = min(self._ema_step, len(self._ema_schedule) - 1)
                loss_dict['ema_tau'] = torch.tensor(self._ema_schedule[idx])

        return pred, loss_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default=f"{Path(__file__).parent.parent}/configs/train_grayscott.yml")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--predictor_path", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run

    cfg.model.objective = "jepa"

    print(OmegaConf.to_yaml(cfg, resolve=True))

    trainer = JepaTrainer(cfg)
    trainer.train()