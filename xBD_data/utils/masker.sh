#!/bin/bash

#SBATCH -n 4
#SBATCH -t 190
#SBATCH -p shared
#SBATCH --mem=10000
#SBATCH -o masker_output_%j.out 
#SBATCH -e masker_errors_%j.err 

python mask_polygons.py --input /n/tambe_lab/disaster_relief/xBD_data/trainn --single-file --border 2
