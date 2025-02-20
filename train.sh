#!/bin/bash

#SBATCH --job-name=InfoGCN_abs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/errors_infogcn_abs.txt'
#SBATCH --output='logs/train_infogcn_abs.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python main.py -p train -l joint_infogcn -r infogcn_abs -d 'Absolute flow window, kernel = 5' -v