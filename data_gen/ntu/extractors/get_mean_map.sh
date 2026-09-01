#!/bin/bash

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

dataset=ntu # (ntu, ntu120)
evaluation=CS # (CS/CV, CSub/CSet)
flow_embedding=cnn # (cnn, base)
flow_type=LK # (RAFT, LK, norm?)
dilation=5
debug=false

# Define file name (may need to change this depending on the dataset)
if [ $flow_embedding == "base" ]; then
    filename=${dataset}_${evaluation}-pose_aligned.npz
    copy_file=./data/${dataset}/aligned_data/pose/${filename}
else
    filename=${dataset}_${evaluation}-poseoff_${flow_type}_D${dilation}_aligned.npz
    copy_file=./data/${dataset}/aligned_data/poseoff/${flow_type}/${filename}
fi

# Check if the file exists... Exit if not!
if [ ! -f $copy_file ]; then
    echo "File ${copy_file} not found!"
    exit 1
fi

# Copy file to jobfs
cp $copy_file ${JOBFS}/data/

data_path="${JOBFS}/data/${filename}"

echo "Dataset: ${dataset}"
echo "Evaluation: ${evaluation}"
echo "Flow embedding: ${flow_embedding}"
echo "Flow type: ${flow_type}"
echo "Dilation: ${dilation}"
echo "Data path: ${data_path}"
echo "Copy file (where will the output of this be saved): ${copy_file}"

args=(
    "-d" "${dataset}"
    "-f" "${flow_embedding}"
    "-e" "${evaluation}"
    "--data_path_overwrite" "${data_path}"
    "--save_path_overwrite" "${copy_file}"
)

# Optionally just debug by only using the first 100 samples and not saving the .npz file
if $debug
then
    args+=("--debug")
fi

# Run the get_mean_map script with the given arguments
srun python data_gen/utils/get_mean_map.py "${args[@]}"
