#!/bin/bash

#SBATCH --gres=gpu  
#SBATCH --gpu-freq=high
#SBATCH -t 0-30:00        
#SBATCH -p gpu            
#SBATCH --mem=40000        
#SBATCH -o myoutput_1_3_extended_0.005lr%j.out 
#SBATCH -e myerrors_1_3_extended_0.005lr%j.err 

#python train.py -x vhr -o '/n/tambe_lab/disaster_relief/multi3net/results/predictions_lovazsh' -f '/n/tambe_lab/disaster_relief/multi3net/results/predictions_1_3/vhr_buildings10m/epoch_09_classes_02.pth' -e 10 -a 0.9 --lr 0.1

python train.py -x vhr -o '/n/tambe_lab/disaster_relief/multi3net/results/predictions_1_3_extended_0.005lr' -f '/n/tambe_lab/disaster_relief/multi3net/results/predictions_1_3/vhr_buildings10m/epoch_19_classes_02.pth' -e 15 -b 8 -a 0.9 --lr 0.005
