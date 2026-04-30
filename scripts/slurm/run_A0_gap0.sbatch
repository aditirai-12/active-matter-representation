#!/bin/bash
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ml9542/active-matter-representation/logs/A0-gap0-%j.out
#SBATCH --error=/scratch/ml9542/active-matter-representation/logs/A0-gap0-%j.err

set -e

cd /scratch/ml9542/active-matter-representation

source /scratch/ml9542/miniconda3/etc/profile.d/conda.sh
conda activate active_matter

export THE_WELL_DATA_DIR=/scratch/ml9542/data
export WANDB_PROJECT=active-matter-representation
export WANDB_ENTITY=aditirai2020-new-york-university
export WANDB_NAME=A0-gap0-${SLURM_JOB_ID}

mkdir -p logs checkpoints dataset_cache

python -m physics_jepa.train_jepa configs/train_physics_jepa_upgraded.yaml \
  dataset.num_frames=8 \
  dataset.gap_frames=0 \
  model.use_channel_stem=false \
  model.use_temporal_mixer=false \
  model.predictor_scale_factor=2 \
  model.predictor_depth=1 \
  train.num_epochs=50 \
  train.batch_size=8 \
  train.distributed=false \
  train.run_name=A0-gap0-${SLURM_JOB_ID}
