#!/bin/bash
#SBATCH --job-name=A12-linear
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ml9542/active-matter-representation/logs/A12-180773-linear-%j.out
#SBATCH --error=/scratch/ml9542/active-matter-representation/logs/A12-180773-linear-%j.err

set -e

cd /scratch/ml9542/active-matter-representation

source /scratch/ml9542/miniconda3/etc/profile.d/conda.sh
conda activate active_matter

export THE_WELL_DATA_DIR=/scratch/ml9542/data
export WANDB_PROJECT=active-matter-representation
export WANDB_ENTITY=aditirai2020-new-york-university
export WANDB_MODE=online
export WANDB_NAME=A12-gap-stem-linear-180773-${SLURM_JOB_ID}

mkdir -p embeddings/A12_gap_stem_180773

A12_ENCODER="/scratch/ml9542/active-matter-representation/checkpoints/active_matter-8frames-cnn-jepa-A12-gap-stem-e50-bs8-180773_2026-05-02_10-24-52/ConvEncoder_49.pth"

python -u -m physics_jepa.finetune \
  --trained_model_path $A12_ENCODER \
  configs/eval_upgraded_linear.yaml \
  dataset.num_frames=8 \
  dataset.gap_frames=8 \
  model.use_channel_stem=true \
  model.use_temporal_mixer=false \
  model.predictor_scale_factor=2 \
  model.predictor_depth=1 \
  ft.embeddings_dir=./embeddings/A12_gap_stem_180773 \
  ft.run_name=A12-gap-stem-linear-180773-${SLURM_JOB_ID}
