#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH -n 2
#SBATCH -t 0-01:00        
#SBATCH --gpu-freq=high
#SBATCH -p gpu_test
#SBATCH --mem-per-gpu=12000
#SBATCH -o myoutput_prime_transform_%j.out 
#SBATCH -e myerrors_prime_transform_%j.err

python train.py -x 'prime_transform' -o '/n/holyscratch01/tambe_lab/disaster_relief/multi3net/results/predictions_prime_transform'  -e 25 -b 10 -c 1
