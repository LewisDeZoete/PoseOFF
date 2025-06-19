#!/bin/bash

#SBATCH --job-name=ucf101_3_avg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=50g
#SBATCH --gres=gpu:1

#SBATCH --error='logs/UCF101/3/error_avg.txt'
#SBATCH --output='logs/UCF101/3/train_avg.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python main.py -c ucf101 -p train -m avg -e 3 -r ucf101_3_avg -d 'UCF-101 eval 3 avg' -v
