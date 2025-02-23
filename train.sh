#!/bin/bash

#SBATCH --job-name=InfoGCN_base
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/errors_infogcn_base.txt'
#SBATCH --output='logs/train_infogcn_base.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python main.py -p train -m base -r infogcn_base -d 'Base model infogcn, no_flow' -v