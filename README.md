# Active Matter Representation Learning
CSCI-GA 2572 Deep Learning — Spring 2026

## Project Overview
Self-supervised representation learning on the active_matter dataset from The Well. The goal is to learn representations of complex physical dynamics and evaluate them via linear probing and kNN regression.

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
- Each trajectory is 81 timesteps; we extract sliding 16-frame windows
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
├── src/
│   ├── dataset.py               # data loading and preprocessing
│   ├── model.py                 # model architecture (TODO)
│   ├── train.py                 # training loop (TODO)
│   └── evaluate.py              # linear probe and kNN evaluation (TODO)
│
├── configs/
│   └── base_config.yaml         # shared training config (TODO)
│
├── data/
│   └── stats/
│       └── train_stats.json     # pre-computed channel + label stats
│
├── scripts/
│   ├── train.slurm              # Slurm job script
│   ├── compute_stats.py         # compute normalization stats from train split
│   ├── create_mock_data.py      # generate mock data for local testing
│   └── test_dataset.py          # validate data pipeline end-to-end
│
├── notebooks/                   # exploratory notebooks
├── ENV.md                       # environment setup instructions
├── requirements.txt             # Python dependencies
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

    sbatch scripts/train.slurm

To check job status:

    squeue -u $USER

---

## Important Rules
- Do NOT use pretrained weights (no ImageNet, CLIP, VideoMAE, etc.)
- Do NOT train on validation or test splits
- Do NOT use labels (alpha, zeta) during representation learning
- Model must be under 100M parameters
- Evaluation must use linear probing and kNN regression only — no MLPs

---