#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH -n 2
#SBATCH -t 0-25:00        
#SBATCH --gpu-freq=high
#SBATCH -p gpu
#SBATCH --mem-per-gpu=12000
#SBATCH -o myoutput_single_fusenet_%j.out 
#SBATCH -e myerrors_single_fusenet_%j.err

python train.py -x 'vhr_pre_post' -o '/n/scratchlfs02/tambe_lab/disaster_relief/multi3net/results/predictions_single_fusenet' -e 25 -b 10
