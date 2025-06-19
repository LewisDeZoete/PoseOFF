#!/bin/bash
#SBATCH --job-name=UCF-101_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=6g

#SBATCH --output='logs/EXTRACT/ucf101/UCF-101_extract.txt'
#SBATCH --error='logs/EXTRACT/ucf101/error_UCF-101_extract.txt'

# Activate the environment
source ../environment/bin/activate

# Main job for each array task
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set."
  exit 1
fi

# Main job for each array task
srun python ./data_gen/ucf101/extractors/${modality}_gendata.py -n $SLURM_ARRAY_TASK_ID