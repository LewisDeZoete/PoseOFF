#!/bin/bash

#SBATCH --job-name=ntu120_CSet_cnn_mean
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00
#SBATCH --mem=20GB
#SBATCH --tmp=350GB

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

date

# Copy data to the Job Filesystem (JOBFS)
mkdir $JOBFS/data

dataset="ntu120" # (ntu, ntu120)
flow_embedding="cnn" # (cnn, base)
evaluation="CSub" # (CS/CV, CSub/CSet)
debug=false

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

echo "Dataset: ${dataset}"
echo "Flow embedding: ${flow_embedding}"
echo "Evaluation: ${evaluation}"
echo "Data path: ${data_path}"

args=(
    "-d" "${dataset}"
    "-f" "${flow_embedding}"
    "-e" "${evaluation}"
)

# Optionally just debug by only using the first 100 samples and not saving the .npz file
if $debug
then
    args+=("--debug")
fi

# Run the get_mean_map script with the given arguments
srun python data_gen/utils/get_mean_map.py "${args[@]}"
