#!/bin/bash
#SBATCH --job-name=NTU_flowpose_seq_transform
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=450g

#SBATCH --output='logs/EXTRACT/NTU_flowpose_seq_transform.txt'
#SBATCH --error='logs/EXTRACT/error_NTU_flowpose_seq_transform.txt'

# Activate the environment
source ../environment/bin/activate

# srun python ./data_gen/infogcn-NTU/get_raw_skes_data.py
# srun python ./data_gen/infogcn-NTU/get_raw_denoised_data.py

# # ~ 8.5 hours, 25g if processing 20000 each time
# srun python ./data_gen/infogcn-NTU/get_flowpose_samples.py 

# Align sequences ~ 2 hours, 450g memory
srun python ./data_gen/infogcn-NTU/seq_transformation.py --flow