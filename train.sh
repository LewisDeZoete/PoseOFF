#!/bin/bash

#SBATCH --job-name=MS_G3D-Train_E80_S45-55-65_G0.1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:1

#SBATCH --error='logs/errors_E80_S45-55-65_G0.1.txt'
#SBATCH --output='logs/train_E80_S45-55-65_G0.1.txt'

#SBATCH --mail-user=ldezoetegrundy@swin.edu.au
#SBATCH --mail-type=ALL

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python new_train.py -p train -r flowpose_cnn_E80_S45-55-65_G0.1 -d 'Epochs: 80\nlr steps: [45, 55, 65]\ngamma at lr steps: 0.1'