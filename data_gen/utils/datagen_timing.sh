#!/bin/bash

#SBATCH --job-name=datagen_timing
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=27g
#SBATCH --output=logs/debug/model_timing/datagen_timing_log_%j.out

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

srun python data_gen/utils/extractors.py
