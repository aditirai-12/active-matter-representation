# Environment Setup

## System
- Platform: NYU HPC Cloud Bursting (OOD)
- CUDA: 11.8
- Python: 3.10

## Setup Instructions

### 1. Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /scratch/$USER/miniconda3
source /scratch/$USER/miniconda3/etc/profile.d/conda.sh
```

### 2. Create and activate environment
```bash
conda create -n active_matter python=3.10 -y
conda activate active_matter
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Notes
- Always activate the environment before running any scripts: `conda activate active_matter`
- Checkpoints are saved to `/scratch/$USER/checkpoints`
- Data is stored at `/scratch/$USER/data/active_matter`
