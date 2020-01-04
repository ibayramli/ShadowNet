#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -t 0-01:00        
#SBATCH --gpu-freq=high
#SBATCH -p gpu_test       
#SBATCH --mem-per-gpu=30000
#SBATCH -o myoutput_prepost_0.01_0.9_8%j.out 
#SBATCH -e myerrors_prepost_0.01_0.9_8%j.err

python train.py -x 'vhr_post_pre' -o '/n/tambe_lab/disaster_relief/multi3net/results/predictions_pre_post' -e 20 -b 1
