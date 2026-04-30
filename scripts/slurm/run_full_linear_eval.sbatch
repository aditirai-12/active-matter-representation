#!/bin/bash
#SBATCH --job-name=full-linear
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=n1s8-t4-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ml9542/active-matter-representation/logs/full-linear-%j.out
#SBATCH --error=/scratch/ml9542/active-matter-representation/logs/full-linear-%j.err

set -e

cd /scratch/ml9542/active-matter-representation
source /scratch/ml9542/miniconda3/etc/profile.d/conda.sh
conda activate active_matter

export THE_WELL_DATA_DIR=/scratch/ml9542/data
export WANDB_PROJECT=active-matter-representation
export WANDB_ENTITY=aditirai2020-new-york-university
export WANDB_MODE=online
export WANDB_NAME=full-linear-${SLURM_JOB_ID}

mkdir -p embeddings/full_A1_A2_A3_A4

FULL_ENCODER="/scratch/ml9542/active-matter-representation/checkpoints/active_matter-8frames-cnn-jepa-A4-full-e50-bs8-177735_2026-04-29_10-33-10/ConvEncoder_49.pth"

python -u -m physics_jepa.finetune \
  --trained_model_path $FULL_ENCODER \
  configs/eval_upgraded_linear.yaml \
  dataset.num_frames=8 \
  dataset.gap_frames=8 \
  model.use_channel_stem=true \
  model.use_temporal_mixer=true \
  model.predictor_scale_factor=3 \
  model.predictor_depth=2 \
  ft.embeddings_dir=./embeddings/full_A1_A2_A3_A4 \
  ft.run_name=full-linear-${SLURM_JOB_ID}
