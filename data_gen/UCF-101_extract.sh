#!/bin/bash
#SBATCH --job-name=UCF-101_extract
#SBATCH --job-name=UCF-101_extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=2000

#SBATCH --output='logs/EXTRACT/UCF-101_extract.txt'
#SBATCH --error='logs/EXTRACT/error_UCF-101_extract.txt'

#SBATCH --array=0-100

# Activate the environment
source ../environment/bin/activate

# Main job for each array task
# srun python ./data_gen/skeleton_gendata.py -n $SLURM_ARRAY_TASK_ID
# srun python ./data_gen/flow_gendata.py -n $SLURM_ARRAY_TASK_ID
srun python ./data_gen/flowpose_gendata.py -n $SLURM_ARRAY_TASK_ID


# Run validation script once all other tasks are finished
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  sbatch --dependency=afterok:$SLURM_ARRAY_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name=UCF-101_validation
#SBATCH --output='logs/EXTRACT/UCF-101_validation.txt'

python ./data_gen/gendata_validation.py
EOF
fi
