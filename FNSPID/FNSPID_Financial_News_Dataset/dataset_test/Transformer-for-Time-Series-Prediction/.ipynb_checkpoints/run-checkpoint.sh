#!/bin/bash

# Request 4 A6000 GPUs for 1 day and 5 hours
#SBATCH --job-name=train
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=spushpit@cs.cmu.edu
#SBATCH --gres=gpu:1     # Request 4 A6000 GPUs
#SBATCH -p gpu 
#SBATCH --time=15:00:00      # Set time to 1 day and 5 hours
#SBATCH --mem=100g            # Adjust memory if needed
#SBATCH --output=final.out   # Save output logs
#SBATCH --error=final.err    # Save error logs

source /home/spushpit/anaconda3/bin/activate
conda activate TVLT

# Run the Python script
python run_stuff.py
