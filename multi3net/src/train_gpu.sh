#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH -t 0-20:00        
#SBATCH --gpu-freq=high
#SBATCH -p gpu
#SBATCH --mem-per-gpu=12000
#SBATCH -o myoutput_single_unet_psp_weight_3_%j.out 
#SBATCH -e myerrors_single_unet_psp_weight_3_%j.err

python train.py -x 'vhr' -o '/n/scratchlfs02/tambe_lab/disaster_relief/multi3net/results/predictions_single_unet_psp_weight_3' -e 20 -b 10
