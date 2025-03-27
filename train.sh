#!/bin/bash

#SBATCH --job-name=nturgbd_CS_avg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=250g
#SBATCH --gres=gpu:1

#SBATCH --error='logs/NTU_RGB_D/CV/errors_avg.txt'
#SBATCH --output='logs/NTU_RGB_D/CV/train_avg.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python main.py -c nturgbd-cross-view -p train -m avg -r nturgbd_CV_avg -d 'nturgbd-cross-view avg' -v