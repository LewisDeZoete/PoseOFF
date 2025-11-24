#!/bin/bash

#SBATCH --job-name=ntu120_CSet_cnn_mean
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=20GB
#SBATCH --tmp=350GB
#SBATCH --error=./logs/EXTRACT/ntu120/error_ntu120_get_mean_map.out
#SBATCH --error=./logs/EXTRACT/ntu120/ntu120_get_mean_map.out

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

date

# Testing copying data to JOBFS during the job
mkdir $JOBFS/data

dataset="ntu120"
flow_embedding="cnn"
evaluation="CSet"

# Define file name (may need to change this depending on the dataset)
if [ $flow_embedding == "base" ]; then
    filename=${dataset}_${evaluation}-pose_aligned.npz
else
    filename=${dataset}_${evaluation}-flowpose_D3_aligned.npz
fi

# Path to file to copy
copy_file=./data/${dataset}/aligned_data/${filename}

# Copy file to jobfs
cp $copy_file ${JOBFS}/data/

data_path="${JOBFS}/data/${filename}"
echo "Data path: ${data_path}"
echo "Dataset: ${dataset}"
echo "Evaluation: ${evaluation}"
echo "Flow embedding: ${flow_embedding}"


srun python feeders/ntu_rgb_d.py -d "${dataset}" -f "${flow_embedding}" -e "${evaluation}" --data_path_overwrite ${data_path}
