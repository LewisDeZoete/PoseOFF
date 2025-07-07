#!/bin/bash

#SBATCH --job-name=nturgbd_CV_cnn_ORIGINAL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --tmp=250GB

#SBATCH --error=error_dataloader_OOM.txt
#SBATCH --output=dataloader_OOM.txt

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

# Testing copying data to JOBFS during the job
mkdir $JOBFS/data

model_type="cnn" # or cnn_TMP
evaluation="CV"

cp ./data/ntu/aligned_data/NTU60_${evaluation}-flowpose_aligned.npz ${JOBFS}/data/
echo $(ls ${JOBFS}/data)

python feeders/ntu_rgb_d.py --data_path_overwrite "${JOBFS}/data" 
