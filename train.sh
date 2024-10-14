#!/bin/bash

#SBATCH --job-name=MS-G3D_train_warmup-exp0.93-retrain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=07:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/errors_train_warmup-exp0.93_SAVE.txt'

#SBATCH --output='logs/train_warmup-exp0.93_SAVE.txt'

#SBATCH --mail-user=ldezoetegrundy@swin.edu.au
#SBATCH --mail-type=ALL

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python new_train.py -p train -s warm10-exp0.93-retrain