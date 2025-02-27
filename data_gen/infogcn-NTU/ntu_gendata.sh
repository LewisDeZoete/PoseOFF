#!/bin/bash
#SBATCH --job-name=NTU_flowpose_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=8g

#SBATCH --output='logs/EXTRACT/NTU_flowpose_extract.txt'
#SBATCH --error='logs/EXTRACT/error_NTU_flowpose_extract.txt'

# Activate the environment
source ../environment/bin/activate

# Main job for each array task
# srun python ./data_gen/infogcn-NTU/get_raw_skes_data.py
# srun python ./data_gen/infogcn-NTU/get_raw_denoised_data.py
srun python ./data_gen/infogcn-NTU/get_flowpose_samples.py # ~ 16-20 hours