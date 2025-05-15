#!/bin/bash

#SBATCH --job-name=ucf101_1_cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=50g
#SBATCH --gres=gpu:1

#SBATCH --error='logs/UCF101/1/error_cnn.txt'
#SBATCH --output='logs/UCF101/1/train_cnn.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python main.py -c ucf101 -p train -m cnn -e 1 -r ucf101_1_cnn -d 'UCF-101 eval 1 cnn TESTING FULL DATASET' -v