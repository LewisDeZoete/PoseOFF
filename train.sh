#!/bin/bash

#SBATCH --job-name=InfoGCN_cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/errors_infogcn_cnn.txt'
#SBATCH --output='logs/train_infogcn_cnn.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python main.py -p train -m cnn -r infogcn_cnn -d 'CNN flow embed, num_cls=1, K=5' -v