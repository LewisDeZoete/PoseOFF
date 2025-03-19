#!/bin/bash

#SBATCH --job-name=nturgbd_CS_base-nstep5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=15g
#SBATCH --gres=gpu:1

#SBATCH --error='logs/NTU_RGB_D/CS/errors_base-nstep5.txt'
#SBATCH --output='logs/NTU_RGB_D/CS/train_base-nstep5.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python main.py -c nturgbd-cross-subject -p train -m base -r nturgbd_CS_base-nstep5 -d 'nturgbd-cross-subject base (n_step=5)' -v