#!/bin/bash
#SBATCH --job-name=A0-linear
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ml9542/active-matter-representation/logs/A0-linear-%j.out
#SBATCH --error=/scratch/ml9542/active-matter-representation/logs/A0-linear-%j.err

set -e

cd /scratch/ml9542/active-matter-representation
source /scratch/ml9542/miniconda3/etc/profile.d/conda.sh
conda activate active_matter

export THE_WELL_DATA_DIR=/scratch/ml9542/data
export WANDB_PROJECT=active-matter-representation
export WANDB_ENTITY=aditirai2020-new-york-university
export WANDB_MODE=online
export WANDB_NAME=A0-linear-${SLURM_JOB_ID}

mkdir -p embeddings/A0

A0_ENCODER="/scratch/ml9542/active-matter-representation/checkpoints/active_matter-8frames-cnn-jepa-A0-gap0-177696_2026-04-29_09-33-37/ConvEncoder_49.pth"

python -u -m physics_jepa.finetune \
  --trained_model_path $A0_ENCODER \
  configs/eval_upgraded_linear.yaml \
  dataset.num_frames=8 \
  dataset.gap_frames=0 \
  model.use_channel_stem=false \
  model.use_temporal_mixer=false \
  model.predictor_scale_factor=2 \
  model.predictor_depth=1 \
  ft.embeddings_dir=./embeddings/A0 \
  ft.run_name=A0-linear-${SLURM_JOB_ID}