#!/bin/bash

#SBATCH --job-name=nturgbd_CV_cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=250g
#SBATCH --gres=gpu:1

#SBATCH --error='logs/NTU_RGB_D/CV/error_cnn_TMP.txt'
#SBATCH --output='logs/NTU_RGB_D/CV/train_cnn_TMP.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python main.py -c nturgbd -p train -m cnn -e CV -r nturgbd_CV_cnn_TMP -d 'nturgbd-cross-view cnn TESTING FULL DATASET' -v
