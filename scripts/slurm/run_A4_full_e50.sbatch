#!/bin/bash
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ml9542/active-matter-representation/logs/A4-full-e50-%j.out
#SBATCH --error=/scratch/ml9542/active-matter-representation/logs/A4-full-e50-%j.err

set -e

cd /scratch/ml9542/active-matter-representation

source /scratch/ml9542/miniconda3/etc/profile.d/conda.sh
conda activate active_matter

export THE_WELL_DATA_DIR=/scratch/ml9542/data
export WANDB_PROJECT=active-matter-representation
export WANDB_ENTITY=aditirai2020-new-york-university
export WANDB_MODE=online
export WANDB_NAME=A4-full-e50-bs8-${SLURM_JOB_ID}

mkdir -p logs checkpoints dataset_cache

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

python -m physics_jepa.train_jepa configs/train_physics_jepa_upgraded.yaml \
  dataset.num_frames=8 \
  dataset.gap_frames=8 \
  model.use_channel_stem=true \
  model.use_temporal_mixer=true \
  model.predictor_scale_factor=3 \
  model.predictor_depth=2 \
  train.num_epochs=50 \
  train.batch_size=8 \
  train.distributed=false \
  train.run_name=A4-full-e50-bs8-${SLURM_JOB_ID}
