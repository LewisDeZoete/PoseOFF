#!/bin/bash

#SBATCH --job-name=UCF-101_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:1

#SBATCH --output='logs/UCF-101_extract.txt'
#SBATCH --error='logs/error_UCF-101_extract.txt'

source ../environment/bin/activate
srun python ./data_gen/flow_gendata.py -n 0