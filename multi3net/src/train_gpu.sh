#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH -n 2
#SBATCH -t 0-30:00        
#SBATCH --gpu-freq=high
#SBATCH -p gpu
#SBATCH --mem-per-gpu=12000
#SBATCH -o myoutput_siam_conc_experimental_%j.out 
#SBATCH -e myerrors_siam_conc_experimental_%j.err

python train.py -x 'pre_post_experimental' -o '/n/holyscratch01/tambe_lab/disaster_relief/multi3net/results/predictions_siam_unet_conc'  -e 25 -b 10
