# Active Matter Representation Learning
CSCI-GA 2572 Deep Learning — Spring 2026

## Project Overview
Self-supervised representation learning on the active_matter dataset from The Well. The goal is to learn representations of complex physical dynamics and evaluate them via linear probing and kNN regression.

## Getting Started

### Note to teammates who don't have the full dataset yet
A mock dataset is provided in data/mock/ for immediate development and testing.
It has the exact same structure as the real dataset but with only 10 samples.
Use it to build and test your code before downloading the full 52GB dataset.

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
- data/mock/              — small mock dataset for development (10 samples, correct shape)
- src/dataset.py          — data loading and preprocessing (P1)
- src/model.py            — model architecture (P2)
- src/train.py            — training loop (P2)
- src/evaluate.py         — linear probe and kNN evaluation (P3)
- configs/base_config.yaml — shared training config
- scripts/train.slurm     — Slurm job script
- notebooks/              — exploratory notebooks
- ENV.md                  — environment setup instructions
- requirements.txt        — Python dependencies
