#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=60g

# Activate the environment
source ../environment/bin/activate

srun python ./data_gen/ntu/concat_poseoff.py --dataset $dataset --dilation $dilation --flow_type $flow_type

# Can remove the flow data...
rm -rf ./data/${dataset}/flow_data/export_tmp
