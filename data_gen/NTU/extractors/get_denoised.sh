#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=10g

source ../environment/bin/activate

srun python ./data_gen/NTU/get_raw_denoised_data.py --dataset $dataset
