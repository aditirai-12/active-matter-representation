#!/bin/bash
#SBATCH --job-name=active_matter_train
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=85G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/%j.out
#SBATCH --error=/scratch/%u/logs/%j.err
#SBATCH --requeue

mkdir -p /scratch/$USER/logs
source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate active_matter
cd /scratch/$USER/active-matter-representation
python src/train.py --config configs/base_config.yaml
