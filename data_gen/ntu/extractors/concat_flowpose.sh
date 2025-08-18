#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=60g

# Activate the environment
source ../environment/bin/activate

srun python ./data_gen/ntu/concat_flowpose.py --dataset $dataset --dilation $dilation

# Can remove the flow data, but it takes so long to reextract...
# rm -rf ./data/${dataset}/flow_data/export_tmp
