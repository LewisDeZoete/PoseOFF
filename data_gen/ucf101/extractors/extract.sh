#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=6g


# Activate the environment
source ../environment/bin/activate

# Main job for each array task
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Error: SLURM_ARRAY_TASK_ID is not set."
  exit 1
fi

# The parameter ${dilation} is only used for poseoff extraction
# The ${flow_type} parameter is for flow extraction, and poseoff
# Main job for each array task
if [[ ${modality} = "poseoff" ]]; then
  srun python ./data_gen/ucf101/gendata/${modality}_gendata.py -n $SLURM_ARRAY_TASK_ID --dilation $dilation --flow_type $flow_type
else
  srun python ./data_gen/ucf101/gendata/${modality}_gendata.py -n $SLURM_ARRAY_TASK_ID --flow_type $flow_type
fi
