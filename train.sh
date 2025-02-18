#!/bin/bash

#SBATCH --job-name=InfoGCN_avg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/errors_infogcn_avg.txt'
#SBATCH --output='logs/train_infogcn_avg.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python main.py -p train -l joint_infogcn -r infogcn_avg -d 'Average flow window, kernel = 9' -v