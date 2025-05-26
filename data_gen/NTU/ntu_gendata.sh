#!/bin/bash
#SBATCH --job-name=NTU120_seq_transform
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=800g

#SBATCH --output='logs/EXTRACT/NTU120/NTU120_seq_transform.txt'
#SBATCH --error='logs/EXTRACT/NTU120/error_NTU120_seq_transform.txt'

# Activate the environment
source ../environment/bin/activate

# # ~ 1.5 hours 
# srun python ./data_gen/NTU/get_raw_skes_data.py --dataset ntu120

# # ~ 20 minutes 6g
# srun python ./data_gen/NTU/get_raw_denoised_data.py --dataset ntu120

# # ~ 8.5 hours, 25g if processing 20000 each time NEEDS CUDA
# srun python ./data_gen/NTU/get_flowpose_samples.py --dataset ntu120 --idx_start 0 --idx_end 20000
# srun python ./data_gen/NTU/get_flowpose_samples.py --dataset ntu120 --idx_start 20000 --idx_end 40000
# srun python ./data_gen/NTU/get_flowpose_samples.py --dataset ntu120 --idx_start 40000 --idx_end 60000

# # TODO: Make this file dependent on the previous three (change from debug)
# srun python ./data_gen/NTU/DEBUG.py
# # Afterwards, delete ./data/ntu[]/flowpose_data/*

# # Align sequences ~ 2 hours, 450g memory for ntu, 900g (or more, haven't fully tested) for ntu120
srun python ./data_gen/NTU/seq_transformation.py --dataset ntu120 --flow