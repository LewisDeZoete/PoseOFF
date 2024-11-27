#!/bin/bash

#SBATCH --job-name=MS_G3D-Train_flowpose-FIXED-cnn_transforms
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=09:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/errors_flowpose-FIXED-cnn_transforms.txt'
#SBATCH --output='logs/train_flowpose-FIXED-cnn_transforms.txt'

#SBATCH --mail-user=ldezoetegrundy@swin.edu.au
#SBATCH --mail-type=ALL

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python new_train.py -p train -r flowpose-cnn -d 'NO TRANSFORMS (oom issues), Save model state every 10 epochs'