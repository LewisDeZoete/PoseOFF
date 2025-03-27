#!/bin/bash
#SBATCH --job-name=NTU120_raw_denoised_skes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=20g

#SBATCH --output='logs/EXTRACT/NTU120_raw_denoised_skes.txt'
#SBATCH --error='logs/EXTRACT/error_NTU120_raw_denoised_skes.txt'

# Activate the environment
source ../environment/bin/activate

# # ~ 1.5 hours 
# srun python ./data_gen/infogcn-NTU/get_raw_skes_data.py --dataset ntu120

# # ~ 20 minutes 6g
# srun python ./data_gen/infogcn-NTU/get_raw_denoised_data.py --dataset ntu120

# # ~ 8.5 hours, 25g if processing 20000 each time
# srun python ./data_gen/infogcn-NTU/get_flowpose_samples.py 

# # Align sequences ~ 2 hours, 450g memory
# srun python ./data_gen/infogcn-NTU/seq_transformation.py --flow

# # TMP
# srun python ./data_gen/infogcn-NTU/DEBUG.py