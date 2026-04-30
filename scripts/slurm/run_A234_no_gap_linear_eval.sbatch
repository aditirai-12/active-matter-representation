#!/bin/bash
#SBATCH --job-name=A234-linear
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=n1s8-t4-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ml9542/active-matter-representation/logs/A234-linear-%j.out
#SBATCH --error=/scratch/ml9542/active-matter-representation/logs/A234-linear-%j.err

set -e

cd /scratch/ml9542/active-matter-representation

source /scratch/ml9542/miniconda3/etc/profile.d/conda.sh
conda activate active_matter

export THE_WELL_DATA_DIR=/scratch/ml9542/data
export WANDB_PROJECT=active-matter-representation
export WANDB_ENTITY=aditirai2020-new-york-university
export WANDB_MODE=online
export WANDB_NAME=A234-no-gap-linear-${SLURM_JOB_ID}

mkdir -p embeddings/A234_no_gap

A234_ENCODER="/scratch/ml9542/active-matter-representation/checkpoints/active_matter-8frames-cnn-jepa-A234-no-gap-e50-bs8-178477_2026-04-29_18-24-18/ConvEncoder_49.pth"

python -u -m physics_jepa.finetune \
  --trained_model_path $A234_ENCODER \
  configs/eval_upgraded_linear.yaml \
  dataset.num_frames=8 \
  dataset.gap_frames=0 \
  model.use_channel_stem=true \
  model.use_temporal_mixer=true \
  model.predictor_scale_factor=3 \
  model.predictor_depth=2 \
  ft.embeddings_dir=./embeddings/A234_no_gap \
  ft.run_name=A234-no-gap-linear-${SLURM_JOB_ID}
