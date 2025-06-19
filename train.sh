#!/bin/bash

#SBATCH --job-name=nturgbd_CV_cnn_ORIGINAL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=9:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --tmp=250GB

#SBATCH --error='logs/NTU_RGB_D/CV/error_cnn_ORIGINAL.txt'
#SBATCH --output='logs/NTU_RGB_D/CV/train_cnn_ORIGINAL.txt'

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

# Testing copying data to JOBFS during the job
mkdir $JOBFS/data

model_type="cnn" # or cnn_TMP
evalulation="CV"

cp "./data/ntu/aligned_data/NTU60_${evalulation}-flowpose_aligned.npz" "$JOBFS/data/"


# -c : config (eg. nturgbd)
# -p : phase (train/test)
# -m : model_type (eg. cnn, cnn, cnn_TMP)
#         ./config/{config}/{phase}_{model_type}.yaml
# -e : evaluation (eg. CV for ntu, CSet for ntu120, 2 for ucf101)
# -r : run_name (eg. nturgbd_CV_cnn_full_flow)
#         checkpoint_file = '{arg.save_location}/{evaluation}/run_name'
# -d : description (eg. 'nturgbd-cross-view cnn full flow')
# -v : verbose

python main.py \
       -c nturgbd \
       -p train \
       -m $model_type \
       -e $evaluation \
       -r nturgbd_CV_ORIGINAL \
       -d 'nturgbd-cross-view full optical flow extraction WITH MASK' \
       --data_path_overwrite $JOBFS/data \
       -v
