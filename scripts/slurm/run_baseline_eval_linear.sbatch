#!/bin/bash
#SBATCH --job-name=baseline-linear-eval
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/ml9542/active-matter-representation/logs/baseline-linear-eval-%j.out
#SBATCH --error=/scratch/ml9542/active-matter-representation/logs/baseline-linear-eval-%j.err

set -e

cd /scratch/ml9542/active-matter-representation

source /scratch/ml9542/miniconda3/etc/profile.d/conda.sh
conda activate active_matter

export THE_WELL_DATA_DIR=/scratch/ml9542/data
export WANDB_PROJECT=active-matter-representation
export WANDB_ENTITY=aditirai2020-new-york-university
export WANDB_MODE=online
export WANDB_NAME=baseline-linear-eval-${SLURM_JOB_ID}

export BASE_ENCODER=$(ls checkpoints/baseline-jepa-full/*/ConvEncoder_3.pth)

echo "Using encoder:"
echo $BASE_ENCODER

python -u -m physics_jepa.finetune configs/eval_baseline_linear.yaml --trained_model_path $BASE_ENCODER
