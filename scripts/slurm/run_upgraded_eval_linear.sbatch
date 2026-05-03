#!/bin/bash
#SBATCH --job-name=upgraded-linear-eval
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/active-matter-representation/logs/upgraded-linear-eval-%j.out
#SBATCH --error=/scratch/%u/active-matter-representation/logs/upgraded-linear-eval-%j.err

set -e

cd /scratch/$USER/active-matter-representation

source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate active_matter

export THE_WELL_DATA_DIR=/scratch/$USER/data
export WANDB_PROJECT=active-matter-representation
export WANDB_ENTITY=aditirai2020-new-york-university
export WANDB_MODE=online
export WANDB_NAME=upgraded-linear-eval-${SLURM_JOB_ID}

# Find the latest upgraded encoder checkpoint
ENCODER_PATH=$(ls -t checkpoints/upgraded-jepa/*/ConvEncoder_*.pth 2>/dev/null | head -1)

if [ -z "$ENCODER_PATH" ]; then
    echo "ERROR: No upgraded encoder checkpoint found in checkpoints/upgraded-jepa/"
    echo "Train the upgraded model first: sbatch scripts/slurm/run_upgraded_jepa.sbatch"
    exit 1
fi

echo "Using encoder: $ENCODER_PATH"

python -u -m physics_jepa.finetune configs/eval_upgraded_linear.yaml --trained_model_path $ENCODER_PATH
