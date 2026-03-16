#!/bin/bash
#
#SBATCH --job-name=model_timing
#SBATCH --mem=15g
#SBATCH --time=01:00:0
#SBATCH --gres=gpu:1
#SBATCH --output=logs/debug/model_timing/slurm_log_%j.out

# # copy_file="./data/ntu/aligned_data/ntu_CS-pose_aligned.npz"
# copy_file="./data/ntu/aligned_data/ntu_CS-flowpose_D3_aligned.npz"

# # Double check that from this script, ${copy_file} exists!
# if [ ! -f $copy_file ]; then
#     echo "File ${copy_file} not found!"
#     exit 1
# fi

# # Create directory on job file system to move dataset to
# mkdir $JOBFS/data

# # Move the dataset (large) to the $JOBFS so it can be memory mapped during training
# echo "Copying file: ${copy_file} to ${JOBFS}/data"
# cp ${copy_file} "${JOBFS}/data/"

echo "========== SLURM JOB INFO =========="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node(s):       $SLURM_NODELIST"
echo "Hostname:     $(hostname)"
echo "CPUs/node:    $SLURM_CPUS_ON_NODE"
echo "GPUs:         $SLURM_GPUS"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "==================================="

echo
echo "CPU INFO:"
lscpu | grep -E 'Model name|Architecture|CPU\(s\)'

echo
echo "GPU INFO:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

source ../environment/bin/activate

models=( "infogcn2" "stgcn2" "msg3d" )
flow_embeddings=( "base" "cnn" )

for model in "${models[@]}"
do
    for flow_embedding in "${flow_embeddings[@]}"
    do
        echo "${model} - ${flow_embedding}"
        srun python training/model_timer.py -m "$model" -f "$flow_embedding"
        echo ""
    done
done
