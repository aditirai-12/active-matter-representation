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

### Step 6 — Generate Mock Data
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

You will see output like this:

    Filesystem      Size  Used  Avail  Use%  Mounted on
    ...             30T   27T   3.8T   88%   /scratch

Look at the Avail column — you need at least 60GB free before downloading.
If you have less than 60GB, contact Person 1 to clear space before proceeding.

Once you have enough space:

    mkdir -p /scratch/$USER/data
    hf auth login
    hf download polymathic-ai/active_matter --repo-type dataset --local-dir /scratch/$USER/data/active_matter

---

## Dataset Structure

### File Organization
- Each .hdf5 file = one unique (alpha, zeta) parameter combination
- Train: 45 files | Valid: 16 files | Test: 21 files
- 3 simulation trajectories per file
- 81 time steps per trajectory
- 256x256 spatial resolution

### The 11 Channels
- Concentration       (t0_fields/concentration) — 1 channel
- Velocity            (t1_fields/velocity)      — 2 channels (x, y)
- Strain-rate tensor  (t2_fields/D)             — 4 channels (2x2)
- Orientation tensor  (t2_fields/E)             — 4 channels (2x2)

### Labels (withheld during training)
- scalars/alpha — active dipole strength (5 discrete values)
- scalars/zeta  — steric alignment (9 discrete values)

### How Samples Are Generated
- 45 files x 3 trajectories = 135 train trajectories
- Sliding 16-frame windows across 81 time steps with 224x224 crops
- ~65 windows per trajectory x 135 = ~8,750 training samples

---

## Repository Structure
```
active-matter-representation/
│
├── src/
│   ├── dataset.py               # data loading and preprocessing
│   ├── model.py                 # model architecture
│   ├── train.py                 # training loop
│   └── evaluate.py              # linear probe and kNN evaluation
│
├── configs/
│   └── base_config.yaml         # shared training config
│
├── scripts/
│   ├── train.slurm              # Slurm job script
│   └── create_mock_data.py      # run this to generate mock data locally
│
├── notebooks/                   # exploratory notebooks
├── ENV.md                       # environment setup instructions
└── requirements.txt             # Python dependencies
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