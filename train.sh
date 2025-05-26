#!/bin/bash

#SBATCH --job-name=nturgbd_CS_cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=14:00:00
#SBATCH --mem-per-cpu=250g
#SBATCH --gres=gpu:1

#SBATCH --error='logs/NTU_RGB_D/CS/error_cnn.txt'
#SBATCH --output='logs/NTU_RGB_D/CS/train_cnn.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python main.py -c nturgbd -p train -m cnn -e CS -r nturgbd_CS_cnn -d 'nturgbd-cross-subject average full dataset' -v
