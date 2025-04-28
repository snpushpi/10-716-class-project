#!/bin/bash

# Request 4 A6000 GPUs for 1 day and 5 hours
#SBATCH --job-name=feature_extraction
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # 1 CPU task per job
#SBATCH --gres=gpu:A6000:2         # 1 GPU (A6000) per job
#SBATCH --time=20:00:00          # 1 day, 23 hours
#SBATCH --mem=128GB                # CPU memory
#SBATCH --output=run1.out
#SBATCH --error=run1.err


source /home/spushpit/anaconda3/bin/activate
conda activate qwenf

python /home/spushpit/FNSPID/inference.py
