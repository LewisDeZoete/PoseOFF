#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --mem-per-cpu=5g

source ../environment/bin/activate

srun python ./data_gen/ntu/get_raw_skes_data.py --dataset $dataset
