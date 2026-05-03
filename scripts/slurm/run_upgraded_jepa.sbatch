#!/bin/bash
#SBATCH --job-name=upgraded-jepa
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/%u/active-matter-representation/logs/upgraded-jepa-%j.out
#SBATCH --error=/scratch/%u/active-matter-representation/logs/upgraded-jepa-%j.err

set -e

cd /scratch/$USER/active-matter-representation

source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
conda activate active_matter

export THE_WELL_DATA_DIR=/scratch/$USER/data
export WANDB_PROJECT=active-matter-representation
export WANDB_ENTITY=aditirai2020-new-york-university
export WANDB_NAME=upgraded-jepa-${SLURM_JOB_ID}
export WANDB_MODE=online
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs checkpoints embeddings dataset_cache

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"

# Auto-detect latest checkpoint for resuming after preemption
CHECKPOINT_DIR="checkpoints/upgraded-jepa"
RESUME_OVERRIDES=""

if [ -d "$CHECKPOINT_DIR" ]; then
    # Find latest epoch number from saved ConvEncoder checkpoints
    LATEST_ENCODER=$(ls -t ${CHECKPOINT_DIR}/*/ConvEncoder_*.pth 2>/dev/null | head -1)

    if [ -n "$LATEST_ENCODER" ]; then
        # Extract epoch number from filename e.g. ConvEncoder_4.pth -> 4
        EPOCH=$(basename $LATEST_ENCODER | grep -oP '\d+(?=\.pth)')
        START_EPOCH=$((EPOCH + 1))

        # Build paths for all checkpoint components
        CHECKPOINT_SUBDIR=$(dirname $LATEST_ENCODER)
        ENCODER_PATH="${CHECKPOINT_SUBDIR}/ConvEncoder_${EPOCH}.pth"
        PREDICTOR_PATH="${CHECKPOINT_SUBDIR}/ConvPredictor_${EPOCH}.pth"
        TARGET_PATH="${CHECKPOINT_SUBDIR}/TargetEncoder_${EPOCH}.pth"

        echo "Resuming from epoch ${EPOCH}, starting at epoch ${START_EPOCH}"
        echo "Encoder:         ${ENCODER_PATH}"
        echo "Predictor:       ${PREDICTOR_PATH}"
        echo "Target encoder:  ${TARGET_PATH}"

        RESUME_OVERRIDES="train.start_epoch=${START_EPOCH} train.encoder_path=${ENCODER_PATH} train.predictor_path=${PREDICTOR_PATH} train.target_encoder_path=${TARGET_PATH}"
    else
        echo "No checkpoints found — starting fresh"
    fi
else
    echo "No checkpoint directory found — starting fresh"
fi

python -m physics_jepa.train_jepa configs/train_physics_jepa_upgraded.yaml $RESUME_OVERRIDES
