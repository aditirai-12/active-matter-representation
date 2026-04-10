# Active Matter Representation Learning
CSCI-GA 2572 Deep Learning — Spring 2026

## Project Overview
Self-supervised representation learning on the active_matter dataset from The Well. The goal is to learn representations of complex physical dynamics and evaluate them via linear probing and kNN regression.

## Getting Started

### Note to teammates who don't have the full dataset yet
Run this script to generate a local mock dataset with the exact same structure as the real data:

    python scripts/create_mock_data.py

The mock files are gitignored and will not be committed to the repo.
Use them to build and test your code before downloading the full 52GB dataset.

### For the full dataset
See ENV.md for setup instructions and download commands.
The full dataset should be downloaded to /scratch/$USER/data/active_matter on the cluster.

## Dataset Structure

### File Organization
- Each .hdf5 file = one unique (alpha, zeta) parameter combination
- Train: 45 files | Valid: 16 files | Test: 21 files

### Inside Each File
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

## Repository Structure
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
└── requirements.txt             # Python dependenciesgit add .gitignore README.md scripts/create_mock_data.py
git commit -m "add gitignore, mock data script, and updated README"
git push