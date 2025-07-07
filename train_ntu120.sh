#!/bin/bash

#SBATCH --job-name=nturgbd120_CSub_base
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --tmp=370GB

#SBATCH --error=logs/NTU_RGB_D120/CSub/error_base.txt
#SBATCH --output=logs/NTU_RGB_D120/CSub/train_base.txt

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

# Create directory on job file system to move dataset to
mkdir $JOBFS/data

model_type="base" # or base, abs...
evaluation="CSub"

# Move the dataset (large) to the $JOBFS so it can be memory mapped during training
if [ "$model_type" = "base" ]; then
    cp ./data/ntu120/aligned_data/NTU120_${evaluation}-pose_aligned.npz ${JOBFS}/data/
elif [ "$model_type" = "abs" ] || [ "$model_type" = "avg" ] || [ "$model_type" = "cnn" ]; then
    cp ./data/ntu120/aligned_data/NTU120_${evaluation}-flowpose_aligned.npz ${JOBFS}/data/
fi

echo $(ls ${JOBFS}/data)


# -c : config (eg. nturgbd120)
# -p : phase (train/test)
# -m : model_type (base, cnn, abs, avg)
#         ./config/{config}/{phase}_{model_type}.yaml
# -e : evaluation (eg. CV for ntu, CSet for ntu120, 2 for ucf101)
# -r : run_name (eg. nturgbd120_CSub_cnn_full_flow)
#         checkpoint_file = '{arg.save_location}/{evaluation}/run_name'
# -d : description (eg. 'nturgbd120-cross-view cnn full flow')
# -v : verbose

python main.py \
       -c nturgbd120 \
       -p train \
       -m $model_type \
       -e "${evaluation}" \
       -r "FULL_FLOW/nturgbd120_CSub_base" \
       -d 'nturgbd120-cross-Subset BASE full optical flow extraction' \
       --data_path_overwrite "${JOBFS}/data" \
       -v
