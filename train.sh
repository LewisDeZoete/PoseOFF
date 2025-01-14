#!/bin/bash

#SBATCH --job-name=MS_G3D-Train_attn_TEST
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/errors_attn_TEST.txt'
#SBATCH --output='logs/train_attn_TEST.txt'

#SBATCH --mail-user=ldezoetegrundy@swin.edu.au
#SBATCH --mail-type=ALL

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python new_train.py -p train -r flowpose_attn_TEST -d 'attention for node embeddings, 2*MS-GCN, Transformer encoder for temporal modelling'