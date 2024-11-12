#!/bin/bash

#SBATCH --job-name=MS_G3D-Train_flowpose-embed_B128_k5_t0.05
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/errors_flowpose-embed_B128_k5_t0.05'
#SBATCH --output='logs/train_flowpose-embed_B128_k5_t0.05'

#SBATCH --mail-user=ldezoetegrundy@swin.edu.au
#SBATCH --mail-type=ALL

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python new_train.py -p train -r flowpose-embed_B128_k5_t0.05