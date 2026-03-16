#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=27g

# Activate the environment
source ../environment/bin/activate

# Print the date and some extract information
date
echo "Extracting $flow_type flow"

# Main job for each array task
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set."
    exit 1
fi

# # ~ 2-12 hours, 25g if processing 2000 each time (depending on flow estimation method)
# Job array submits a batch of 2000 from
# ( $SLURM_ARRAY_ID-2000, $SLURM_ARRAY_ID-1 )
case $flow_type in
    "LK")
        srun python ./data_gen/ntu/get_poseoff_samples_LK.py --dataset $dataset --batch_number $SLURM_ARRAY_TASK_ID --dilation $dilation;;
    *)
        srun python ./data_gen/ntu/get_poseoff_samples.py --dataset $dataset --batch_number $SLURM_ARRAY_TASK_ID --dilation $dilation;;
esac
