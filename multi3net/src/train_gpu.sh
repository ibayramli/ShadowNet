#!/bin/bash

#SBATCH --gres=gpu  
#SBATCH --gpu-freq=high
#SBATCH -t 0-30:00        
#SBATCH -p gpu            
#SBATCH --mem=40000        
#SBATCH -o myoutput_prepost_0.01_0.9_8%j.out 
#SBATCH -e myerrors_prepost_0.01_0.9_8%j.err

python train.py -x 'vhr_post_pre' -o '/n/tambe_lab/disaster_relief/multi3net/results/predictions_pre_post' -e 20 -b 8 
