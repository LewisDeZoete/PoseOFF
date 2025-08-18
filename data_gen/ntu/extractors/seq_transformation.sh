#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --mem=800g

source ../environment/bin/activate

echo "Dilation: $dilation"

# Use --split at the end to re-split the dataset
# NOTE: This does not produce the aligned dataset (due to OOM issues), you will need to run it again

srun python ./data_gen/ntu/seq_transformation.py --dataset $dataset --mod "_D${dilation}" --flow --split
srun python ./data_gen/ntu/seq_transformation.py --dataset $dataset --mod "_D${dilation}" --flow
