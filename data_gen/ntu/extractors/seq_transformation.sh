#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=04:00:00
#SBATCH --mem=800g

source ../environment/bin/activate

echo "Dataset: $dataset"
echo "Dilation: $dilation"
echo "Flow type: $flow_type"

srun python ./data_gen/ntu/seq_transformation.py --dataset $dataset --dilation $dilation --flow --flow_type $flow_type --split
srun python ./data_gen/ntu/seq_transformation.py --dataset $dataset --dilation $dilation --flow --flow_type $flow_type
