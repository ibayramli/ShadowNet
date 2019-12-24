#!/bin/bash

#SBATCH --gres=gpu
#SBATCH -n 1
#SBATCH --gpu-freq=high
#SBATCH -t 0-06:00
#SBATCH -p gpu
#SBATCH --mem=8000
#SBATCH -o best_epoch_output_%j.out
#SBATCH -e best_epoch_errors_%j.err

python best_epoch.py
