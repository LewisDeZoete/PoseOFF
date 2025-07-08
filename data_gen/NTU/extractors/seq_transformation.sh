#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --mem=450g

source ../environment/bin/activate

srun python ./data_gen/NTU/seq_transformation.py --dataset $dataset --flow
