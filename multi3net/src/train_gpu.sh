#!/bin/bash

#SBATCH --gres=gpu  
#SBATCH --gpu-freq=high
#SBATCH -t 0-30:00        
#SBATCH -p gpu            
#SBATCH --mem=40000        
#SBATCH -o myoutput_1_3_%j.out 
#SBATCH -e myerrors_1_3_%j.err 

python train.py -x vhr -o '/n/tambe_lab/disaster_relief/multi3net/results/predictions_1_3' -e 20
