#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=27g

# Activate the environment
source ../environment/bin/activate

# Main job for each array task
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set."
    exit 1
fi

# # ~ 12 hours, 25g if processing 2000 each time NEEDS CUDA
# Job array submits a batch of 2000 from
# ( $SLURM_ARRAY_ID-2000, $SLURM_ARRAY_ID-1 )
srun python ./data_gen/NTU/get_flowpose_samples.py --dataset $dataset --batch_number $SLURM_ARRAY_TASK_ID
