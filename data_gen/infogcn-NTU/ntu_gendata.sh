#!/bin/bash
#SBATCH --job-name=NTU120_seq_transform
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=450g

#SBATCH --output='logs/EXTRACT/NTU120/NTU120_seq_transform.txt'
#SBATCH --error='logs/EXTRACT/NTU120/error_NTU120_seq_transform.txt'

# Activate the environment
source ../environment/bin/activate

# # ~ 1.5 hours 
# srun python ./data_gen/infogcn-NTU/get_raw_skes_data.py --dataset ntu120

# # ~ 20 minutes 6g
# srun python ./data_gen/infogcn-NTU/get_raw_denoised_data.py --dataset ntu120

# # ~ 8.5 hours, 25g if processing 20000 each time NEEDS CUDA
# srun python ./data_gen/infogcn-NTU/get_flowpose_samples.py --dataset ntu120 --idx_start 0 --idx_end 20000
# srun python ./data_gen/infogcn-NTU/get_flowpose_samples.py --dataset ntu120 --idx_start 20000 --idx_end 40000
# srun python ./data_gen/infogcn-NTU/get_flowpose_samples.py --dataset ntu120 --idx_start 40000 --idx_end 60000

# # Align sequences ~ 2 hours, 450g memory
srun python ./data_gen/infogcn-NTU/seq_transformation.py --dataset ntu120 --flow --realign

# # TMP
# srun python ./data_gen/infogcn-NTU/DEBUG.py