#!/bin/bash

#SBATCH --job-name=InfoGCN_TEST
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/errors_infogcn_TEST.txt'
#SBATCH --output='logs/train_infogcn_TEST.txt'

#SBATCH --mail-user=ldezoetegrundy@swin.edu.au
#SBATCH --mail-type=ALL

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python tester.py -p train -l joint_infogcn -r infogcn_TEST -d 'First test run for infogcn full training'