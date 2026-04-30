# Active Matter Representation Learning
CSCI-GA 2572 Deep Learning — Spring 2026

## Project Overview
Self-supervised representation learning on the active_matter dataset from The Well. The goal is to learn representations of complex physical dynamics and evaluate them via linear probing and kNN regression.

---

## What Changed (trying-new-model branch)

This branch contains significant architectural improvements over the baseline JEPA. Here's a summary of every change and why it was made.

### 1. EMA Target Encoder
**Files:** `physics_jepa/train_jepa.py`, `physics_jepa/train.py`

The baseline encoded both context and target frames through the same encoder with gradients flowing through both branches. This forces VICReg to do all the collapse prevention work alone. Now the target branch uses a separate encoder updated via exponential moving average (EMA) with stop-gradient — the standard approach in V-JEPA, I-JEPA, BYOL, and DINO.

- Target encoder is a deep copy of the online encoder, updated each step: `p_target = τ * p_target + (1-τ) * p_online`
- Momentum τ follows a cosine schedule from 0.996 → 1.0 over training
- Configurable via `ema_start` and `ema_end` in the train config
- Target encoder checkpoints are saved alongside regular checkpoints as `TargetEncoder_{epoch}.pth`
- Lifecycle hooks added to `Trainer` base class: `on_training_start()`, `on_after_optimizer_step()`, `save_extra_state()`

### 2. Wider Channel Dimensions
**Files:** config changes only

Baseline dims `[16, 32, 64, 128, 128]` → upgraded dims `[32, 64, 128, 256, 256]`. The narrow 128-channel bottleneck couldn't represent the full dynamics of 11 physical fields (especially the 8 tensor channels that drive ζ variation). The wider network is still well under the 100M parameter limit.

### 3. Temporal Preservation in Encoder
**Files:** `physics_jepa/utils/model_utils.py`, `physics_jepa/model.py`, `physics_jepa/finetuner.py`, `scripts/eval_knn.py`

With 16-frame input, the encoder used to squeeze time as 16→8→4→2→1 and drop the time axis entirely. The predictor then operated on a purely spatial 2D feature map with no temporal structure — it couldn't reason about *how* features evolve, only what they look like at one instant.

With `preserve_temporal: true`, the last downsample uses stride `(1,2,2)` instead of `(2,2,2)`, keeping time as 16→8→4→2→2. The output stays 5D as `(B, C, 2, H, W)`. The predictor now uses 3D convolutions and can see temporal evolution patterns, which is critical for distinguishing ζ values (steric alignment manifests through orientation dynamics over time).

- Controlled by `preserve_temporal: true` in model config (defaults to `false` for backward compatibility)
- Predictor automatically switches to Conv3d when enabled
- Eval code handles both 4D and 5D embeddings via generalized average pooling

### 4. Per-Channel Normalization (fixed)
**Files:** `src/dataset.py`, `physics_jepa/data.py`

The normalization code existed but was using placeholder stats (mean=0, std=1 for all channels — effectively a no-op). Now uses real stats from `data/stats/train_stats.json`. This matters especially for the concentration channel, which has std ≈ 0.003 vs velocity channels at std ≈ 0.57 — a 200× scale difference.

### 5. Rebalanced Loss Coefficients
**Files:** config changes only

Baseline used `sim=2, std=40, cov=2` — the variance penalty dominated, which made sense when VICReg alone was preventing collapse. With EMA handling collapse structurally, we rebalanced to `sim=10, std=10, cov=1` so the similarity (prediction quality) term gets more weight.

### 6. Attentive Pooling Removed
**Files:** `physics_jepa/finetuner.py`, `physics_jepa/train.py`, all configs

Attentive pooling is a complex evaluation head (cross-attention + MLP), which is prohibited by the project rules. Removed all `use_attentive_pooling` branches and config keys to prevent accidental use. Only `RegressionHead` (single linear layer) and kNN are used for evaluation.

---

## Configs

| Config | Purpose |
|--------|---------|
| `configs/train_baseline.yaml` | Train the pre-improvement model (narrow dims, no EMA, old loss) |
| `configs/eval_baseline_linear.yaml` | Linear probe eval for baseline checkpoints |
| `configs/train_physics_jepa_upgraded.yaml` | Train the improved model (all changes above) |
| `configs/eval_upgraded_linear.yaml` | Linear probe eval for upgraded checkpoints |

The `train_physics_jepa.yaml` config is an intermediate version — ignore it or delete it.

---

## Quickstart (for Teammates!!)

### Step 1 — Access the Cluster
1. Install Cisco AnyConnect VPN from [here](https://support.nyu.edu/esc?id=kb_article&sysparm_article=KB0011609&table=kb_knowledge&searchTerm=VPN)
2. Connect to vpn.nyu.edu using your NYU NetID and password
3. Go to https://ood-burst-001.hpc.nyu.edu in your browser
4. Log in with your NYU credentials
5. Click Interactive Apps → Code Server
6. Fill in the form:
   - Slurm Account: csci_ga_2572-2026sp
   - Slurm Partition: n2c48m24
   - Number of GPUs: 0
   - Working directory: /scratch/$USER
   - Number of hours: 2
7. Click Launch, wait for it to go green, then click Connect to Code Server

### Step 2 — Set Up Your Environment
Run these commands in the terminal one at a time:

    cd /scratch/$USER
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /scratch/$USER/miniconda3
    source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
    conda create -n active_matter python=3.10 -y
    conda activate active_matter

Note: you will need to run these two lines every time you open a new terminal:

    source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
    conda activate active_matter

### Step 3 — Get a GitHub Personal Access Token
You need this before cloning the repo:
1. Go to github.com → click your profile picture → Settings
2. Scroll down to Developer settings (bottom of left sidebar)
3. Click Personal access tokens → Tokens (classic)
4. Click Generate new token (classic)
5. Name it nyu-cluster, set expiration to 90 days, check the repo scope
6. Click Generate token and copy it immediately — you will not see it again

### Step 4 — Clone the Repo

    cd /scratch/$USER
    git clone https://YOUR_TOKEN@github.com/aditirai-12/active-matter-representation.git
    cd active-matter-representation

Replace YOUR_TOKEN with the token you copied in Step 3.

### Step 5 — Install Dependencies

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt

### Step 6 — Generate Mock Data (optional, for offline development)
Run this to generate a local mock dataset with the exact same structure as the real data:

    python scripts/create_mock_data.py

The mock files are gitignored and will not be committed to the repo.
Use them to build and test your code before downloading the full 52GB dataset.

### Step 7 — Set Up Weights & Biases
    wandb login

To get your API key:
1. Go to wandb.ai and sign in or create an account
2. Click your profile picture → Settings
3. Scroll down to API keys → click New key
4. Name it nyu-cluster and copy the key
5. Paste it into the terminal when prompted

If you don't have access, ask Aditi to add you to the W&B project as a member.

### Step 8 — Download the Full Dataset (only when ready to train)
First check your available storage by running:

    df -h /scratch/$USER

You need at least 60GB free before downloading.

Once you have enough space:

    mkdir -p /scratch/$USER/data
    hf auth login
    hf download polymathic-ai/active_matter --repo-type dataset --local-dir /scratch/$USER/data/active_matter

### Step 9 — Verify the Data Pipeline

    python scripts/test_dataset.py --root /scratch/$USER/data/active_matter --stats data/stats/train_stats.json

This runs a full check: split sizes, SSL/eval modes, normalization, determinism, and DataLoader batching. All checks should pass before you start training.

---

## Training & Evaluation

### Train the upgraded model

    export THE_WELL_DATA_DIR=/scratch/$USER/data
    python -m physics_jepa.train_jepa configs/train_physics_jepa_upgraded.yaml

Or submit via slurm:

    sbatch scripts/slurm/run_baseline_jepa.sbatch

(Update the sbatch script to point to `train_physics_jepa_upgraded.yaml` and your scratch path first.)

### Run linear probe evaluation

    python -u -m physics_jepa.finetune configs/eval_upgraded_linear.yaml --trained_model_path <path_to_ConvEncoder_checkpoint.pth>

### Run kNN evaluation

    python scripts/eval_knn.py --embeddings_dir ./embeddings/upgraded --output_csv results/upgraded_knn_val.csv

---

## Dataset Structure

### File Organization
After downloading, data lives at:

    /scratch/$USER/data/active_matter/data/
    ├── train/    45 .hdf5 files
    ├── valid/    16 .hdf5 files
    └── test/     21 .hdf5 files

Each .hdf5 file = one unique (alpha, zeta) parameter combination.

### Trajectories
- 5 trajectories per parameter set (per the HuggingFace README)
- 175 total training trajectories (45 files × ~3.9 avg, since some combos go to valid/test)
- 81 time steps per trajectory
- 256x256 spatial resolution (cropped to 224x224 for training)

### The 11 Channels
| Index | Field | Source |
|-------|-------|--------|
| 0 | Concentration | t0_fields/concentration — 1 channel |
| 1–2 | Velocity (x, y) | t1_fields/velocity — 2 channels |
| 3–6 | Strain-rate tensor D | t2_fields/D — 4 channels (2×2 flattened) |
| 7–10 | Orientation tensor E | t2_fields/E — 4 channels (2×2 flattened) |

### Labels (withheld during SSL training)
- scalars/alpha — active dipole strength (5 discrete values: -1, -2, -3, -4, -5)
- scalars/zeta — steric alignment (9 discrete values: 1, 3, 5, 7, 9, 11, 13, 15, 17)
- 45 unique (alpha, zeta) combinations total

### How Training Samples Are Generated
- Each trajectory is 81 timesteps; we extract sliding windows of 2×num_frames + gap_frames
- Context = first num_frames frames, target = last num_frames frames (with gap in between)
- Random spatial crop 256→224 during training; center crop during eval
- Random H/V flips during training; no flips during eval

### Pre-computed Statistics
Channel and label normalization stats are saved in `data/stats/train_stats.json`. These were computed from the full training split only (no data leakage). To recompute:

    python scripts/compute_stats.py --root /scratch/$USER/data/active_matter --split train --output data/stats/train_stats.json

---

## Repository Structure
```
active-matter-representation/
│
├── physics_jepa/                # Main codebase
│   ├── train.py                 # Base Trainer class with lifecycle hooks
│   ├── train_jepa.py            # JepaTrainer — EMA target encoder, JEPA forward pass
│   ├── model.py                 # Model factory (get_model_and_loss_cnn)
│   ├── finetuner.py             # Linear probe evaluation pipeline
│   ├── finetune.py              # Finetuner entry point
│   ├── data.py                  # Data loading for JEPA (context/target pairs)
│   ├── videomae.py              # VideoMAE architecture (alternative approach)
│   ├── attentive_pooler.py      # Attentive pooling (used by baselines only)
│   ├── baselines/               # Baseline model implementations
│   └── utils/
│       ├── model_utils.py       # ConvEncoder, ConvPredictor, schedulers
│       ├── model_summary.py     # Parameter counting utilities
│       ├── data_utils.py        # Normalization helpers
│       ├── train_utils.py       # DDP setup, loss gathering
│       └── hydra.py             # Config composition
│
├── src/
│   └── dataset.py               # ActiveMatterDataset with per-channel normalization
│
├── configs/
│   ├── train_baseline.yaml              # Baseline JEPA training
│   ├── eval_baseline_linear.yaml        # Baseline linear probe eval
│   ├── train_physics_jepa_upgraded.yaml  # Upgraded JEPA training (all improvements)
│   ├── eval_upgraded_linear.yaml        # Upgraded linear probe eval
│   ├── dataset/                         # Dataset-specific defaults
│   ├── model/                           # Model size presets
│   └── ft/                              # Finetuning presets (linear, mlp)
│
├── data/
│   └── stats/train_stats.json   # Pre-computed channel + label stats
│
├── scripts/
│   ├── eval_knn.py              # kNN regression evaluation
│   ├── collapse_check.py        # Check for representation collapse
│   ├── compute_stats.py         # Compute normalization stats
│   ├── create_mock_data.py      # Generate mock data for testing
│   ├── test_dataset.py          # Validate data pipeline
│   └── slurm/                   # Slurm job scripts
│
├── results/                     # Evaluation results (CSV)
├── ENV.md
├── requirements.txt
└── README.md
```

---

## Compute Resources
- Account: csci_ga_2572-2026sp
- Quota: 300 GPU hours per student (900 total across the team)
- Recommended partition for training: c12m85-a100-1 (1 A100 40GB GPU)
- Recommended partition for setup/debugging: n2c48m24 (CPU only)

To check your remaining GPU quota:

    sacct -A csci_ga_2572-2026sp --format=JobID,Elapsed,AllocGRES

To submit a training job:

    sbatch scripts/slurm/run_baseline_jepa.sbatch

To check job status:

    squeue -u $USER

**Important:** Cloud resources run on spot instances and may be preempted. The EMA target encoder checkpoint is saved alongside regular checkpoints. To resume, set `target_encoder_path` in the train config.

---

## Important Rules
- Do NOT use pretrained weights (no ImageNet, CLIP, VideoMAE, etc.)
- Do NOT train on validation or test splits
- Do NOT use labels (alpha, zeta) during representation learning
- Model must be under 100M parameters
- Evaluation must use linear probing and kNN regression only — no MLPs, no attentive pooling