#!/bin/bash

#SBATCH --gres=gpu 
#SBATCH -n 1  
#SBATCH --gpu-freq=high
#SBATCH -t 0-01:00        
#SBATCH -p gpu_test
#SBATCH --mem=8000        
#SBATCH -o predict_output_submission_%j.out 
#SBATCH -e predict_error_submission_%j.err 

python predict.py -p '/n/tambe_lab/disaster_relief/xBD_data/xview_test' -o '/n/tambe_lab/disaster_relief/multi3net/src' -e 19 -b 1
