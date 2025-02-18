#!/bin/bash
#SBATCH --job-name=NTU_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=24g

#SBATCH --output='logs/EXTRACT/NTU_extract.txt'
#SBATCH --error='logs/EXTRACT/error_NTU_extract.txt'

# Activate the environment
source ../environment/bin/activate

# Main job for each array task
srun python ./data_gen/MS-G3D-NTU/ntu_gendata.py --n_cores 64