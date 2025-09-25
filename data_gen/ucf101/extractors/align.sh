#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --mem=250g

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

python data_gen/ucf101/seq_transformation.py --dilation $dilation
