#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH -n 2
#SBATCH -t 0-01:00        
#SBATCH --gpu-freq=high
#SBATCH -p gpu_test
#SBATCH --mem-per-gpu=12000
#SBATCH -o myoutput_siam_unet_conc_val_%j.out 
#SBATCH -e myerrors_siam_unet_conc_val_%j.err

python train.py -x 'pre_post' -o '.' -f '/n/holyscratch01/tambe_lab/disaster_relief/multi3net/results/predictions_siam_unet_conc/vhr_pre_post_buildings10m/epoch_24_classes_02.pth'  -e 1 -b 5
