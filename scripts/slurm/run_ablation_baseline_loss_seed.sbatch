#!/bin/bash
#SBATCH --job-name=abl-loss-s
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/active-matter-representation/logs/abl-baseline-loss-seed-%j.out
#SBATCH --error=/scratch/%u/active-matter-representation/logs/abl-baseline-loss-seed-%j.err

cd /scratch/$USER/active-matter-representation
source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate active_matter
export THE_WELL_DATA_DIR=/scratch/$USER/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m physics_jepa.train_jepa configs/train_physics_jepa_upgraded.yaml \
  train.sim_coeff=2 train.std_coeff=40 train.cov_coeff=2 \
  train.run_name=ablation-baseline-loss-seed \
  out_path=./checkpoints/ablation-baseline-loss-seed
