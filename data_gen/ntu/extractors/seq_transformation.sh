#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --mem=800g

source ../environment/bin/activate

echo "Dilation: $dilation"

srun python ./data_gen/ntu/seq_transformation.py --dataset $dataset --mod "_D${dilation}" --flow --split
srun python ./data_gen/ntu/seq_transformation.py --dataset $dataset --mod "_D${dilation}" --flow
