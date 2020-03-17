#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH -n 2
#SBATCH -t 0-15:00        
#SBATCH --gpu-freq=high
#SBATCH -p gpu
#SBATCH --mem-per-gpu=12000
#SBATCH -o myoutput_unet_basic_extra_%j.out 
#SBATCH -e myerrors_unet_basic_extra_%j.err

python train.py -x 'post' -o '/n/holyscratch01/tambe_lab/disaster_relief/multi3net/results/predictions_single_unet_basic_weight_3/vhr_buildings10m_extra_train' -f '/n/holyscratch01/tambe_lab/disaster_relief/multi3net/results/predictions_single_unet_basic_weight_3/vhr_buildings10m/epoch_20_classes_02.pth' -e 5 -b 10
