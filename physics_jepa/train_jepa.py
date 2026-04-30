import argparse
from pathlib import Path
from omegaconf import OmegaConf

from .train import Trainer
from .utils.hydra import compose

class JepaTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def pred_fn(self, batch, model_components, loss_fn):
        encoder, predictor = model_components
        # print("context shape before encoder:", batch["context"].shape)
        # print("target shape before encoder:", batch["target"].shape)
        # print("gap test mean abs diff:", (batch["context"] - batch["target"]).abs().mean().item())
        ctx_embed = encoder(batch['context'])
        tgt_embed = encoder(batch['target'])
        pred = predictor(ctx_embed)
        
        # Compute loss on projected embeddings
        if len(pred.shape) < 5:
            loss_dict = loss_fn(pred.unsqueeze(2), tgt_embed.unsqueeze(2))
        else:
            loss_dict = loss_fn(pred, tgt_embed)

        # representation collapse check
        with torch.no_grad():
            # std across batch dimension per feature
            ctx_std = ctx_embed.float().std(dim=0).mean()
            tgt_std = tgt_embed.float().std(dim=0).mean()
            loss_dict['ctx_embed_std'] = ctx_std
            loss_dict['tgt_embed_std'] = tgt_std

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
    # cfg.train.encoder_path = args.encoder_path
    # cfg.train.predictor_path = args.predictor_path
    
    cfg.model.objective = "jepa"

    print(OmegaConf.to_yaml(cfg, resolve=True))

    trainer = JepaTrainer(cfg)
    trainer.train()