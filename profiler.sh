#!/bin/bash

#SBATCH --job-name=train_profiling
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:40:00
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/err_profiling.txt'
#SBATCH --output='logs/profiling.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python ./training/train_infogcn.py

rm DELETE_ME.pt