#!/bin/bash
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ml9542/active-matter-representation/logs/baseline-jepa-full-%j.out
#SBATCH --error=/scratch/ml9542/active-matter-representation/logs/baseline-jepa-full-%j.err

set -e

cd /scratch/ml9542/active-matter-representation

source /scratch/ml9542/miniconda3/etc/profile.d/conda.sh
conda activate active_matter

export THE_WELL_DATA_DIR=/scratch/ml9542/data
export WANDB_PROJECT=active-matter-representation
export WANDB_ENTITY=aditirai2020-new-york-university
export WANDB_NAME=baseline-jepa-full-${SLURM_JOB_ID}
export WANDB_MODE=online

mkdir -p logs checkpoints dataset_cache

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

python -m physics_jepa.train_jepa configs/train_baseline.yaml
