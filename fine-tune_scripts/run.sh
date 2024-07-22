#!/bin/bash

#SBATCH -n 4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --job-name=train
#SBATCH --output=train-0.out
#SBATCH --error=train-0.err

# Activate virtual environment
source /cluster/home/lnonino/miniconda3/bin/activate multi-CLIP

# Check GPU
python -c 'import torch; print("torch version:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA version:", torch.version.cuda)'

# Run Python script
python /cluster/home/lnonino/Making-CLIP-features-multiview-consistent/fine-tune_scripts/contrastive-fine-tune.py
