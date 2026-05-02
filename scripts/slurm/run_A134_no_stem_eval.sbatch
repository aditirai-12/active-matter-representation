#!/bin/bash
#SBATCH --job-name=A134-no-stem-linear
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=n1s8-t4-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/nss9923/active-matter-representation/logs/A134-linear-%j.out
#SBATCH --error=/scratch/nss9923/active-matter-representation/logs/A134-linear-%j.err

set -e

cd /scratch/nss9923/active-matter-representation
source /scratch/nss9923/miniconda3/etc/profile.d/conda.sh
conda activate active_matter

export THE_WELL_DATA_DIR=/scratch/nss9923/data
export WANDB_PROJECT=active-matter-representation
export WANDB_ENTITY=aditirai2020-new-york-university
export WANDB_MODE=online
export WANDB_NAME=A134-no-stem-linear-${SLURM_JOB_ID}

mkdir -p embeddings/A134_no_stem

A134_ENCODER="/scratch/nss9923/active-matter-representation/checkpoints/active_matter-8frames-cnn-jepa-ablation-no-stem_2026-04-29_18-53-16/ConvEncoder_49.pth"

python -u -m physics_jepa.finetune \
  --trained_model_path $A134_ENCODER \
  configs/eval_upgraded_linear.yaml \
  dataset.num_frames=8 \
  dataset.gap_frames=8 \
  model.use_channel_stem=false \
  model.use_temporal_mixer=true \
  model.predictor_scale_factor=3 \
  model.predictor_depth=2 \
  ft.embeddings_dir=./embeddings/A134_no_stem \
  ft.run_name=A134-no-stem-linear-${SLURM_JOB_ID}