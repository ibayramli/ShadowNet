#!/bin/bash
#SBATCH --gres=gpu
#SBATCH -t 0-20:00        
#SBATCH --gpu-freq=high
#SBATCH -p gpu
#SBATCH --mem=20000
#SBATCH -o myoutput_prepost_pspnet_%j.out 
#SBATCH -e myerrors_prepost_single_pspnet_%j.err

python train.py -x 'vhr_post_pre' -o '/n/tambe_lab/disaster_relief/multi3net/results/predictions_pspnet_post_spacenet' -e 20 -b 10
