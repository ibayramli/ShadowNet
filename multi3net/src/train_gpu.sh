#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH -t 0-01:00        
#SBATCH --gpu-freq=high
#SBATCH -p gpu_test
#SBATCH --mem-per-gpu=20000
#SBATCH -o myoutput_single_unet_encoded_%j.out 
#SBATCH -e myerrors_single_unet_encoded_%j.err

python train.py -x 'vhr' -o '/n/tambe_lab/disaster_relief/multi3net/results/predictions_single_unet_encoded' -e 20 -b 5
