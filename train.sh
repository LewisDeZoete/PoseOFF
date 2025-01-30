#!/bin/bash

#SBATCH --job-name=InfoGCN_abs_window_mean_flow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/errors_infogcn_abs_window_mean_flow.txt'
#SBATCH --output='logs/train_infogcn_abs_window_mean_flow.txt'

#SBATCH --mail-user=ldezoetegrundy@swin.edu.au
#SBATCH --mail-type=ALL

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python tester.py -p train -l joint_infogcn -r infogcn_abs_window_mean_flow -d 'Absolute window mean flow (channels=4), Linear embed (default)' -v