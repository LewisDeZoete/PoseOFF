#!/bin/bash

#SBATCH --job-name=nturgbd_CS_base_TMP
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=5GB
#SBATCH --gres=gpu:1
#SBATCH --tmp=200GB

#SBATCH --error=logs/NTU_RGB_D/CS/FULL_FLOW/error_base_TMP.txt
#SBATCH --output=logs/NTU_RGB_D/CS/FULL_FLOW/train_base_TMP.txt

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

# Create directory on job file system to move dataset to
mkdir $JOBFS/data

model_type="base_TMP" # or base, abs...
evaluation="CS"

# Move the dataset (large) to the $JOBFS so it can be memory mapped during training
if [ "$model_type" = "base" ]; then
    cp ./data/ntu/aligned_data/NTU60_${evaluation}-pose_aligned.npz ${JOBFS}/data/
elif [ "$model_type" = "abs" ] || [ "$model_type" = "avg" ] || [ "$model_type" = "cnn" ] || ["$model_type" = "base_TMP"]; then
    # !!!!!!!!REMOVE THE BASE_TMP OPTION HERE!!!!!!!!!!
    cp ./data/ntu/aligned_data/NTU60_${evaluation}-flowpose_aligned.npz ${JOBFS}/data/
fi

echo $(ls ${JOBFS}/data)


# -c : config (eg. nturgbd)
# -p : phase (train/test)
# -m : model_type (base, cnn, abs, avg)
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
       -e "${evaluation}" \
       -r "FULL_FLOW/nturgbd_CS_base_TMP" \
       -d 'nturgbd-cross-subject BASE full optical flow extraction' \
       --data_path_overwrite "${JOBFS}/data" \
       -v
