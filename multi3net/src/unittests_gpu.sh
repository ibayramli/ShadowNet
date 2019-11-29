#!/bin/bash

#SBATCH --gres=gpu  
#SBATCH --gpu-freq=high
#SBATCH -t 0-07:00        
#SBATCH -p gpu            
#SBATCH --mem=10000        
#SBATCH -o testoutput_%j.out 
#SBATCH -e testerrors_%j.err 

python unittests.py
