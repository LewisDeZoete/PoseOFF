#!/bin/bash

#SBATCH --job-name=UCF-101_align
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=250g

#SBATCH --error='logs/EXTRACT/ucf101/error_UCF-101_align.txt'
#SBATCH --output='logs/EXTRACT/ucf101/UCF-101_align.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python data_gen/ucf101/seq_transformation.py