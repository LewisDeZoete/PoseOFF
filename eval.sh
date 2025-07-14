#!/bin/bash

#SBATCH --job-name=eval_CS_cnn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --mem=15GB
#SBATCH --gres=gpu:1
#SBATCH --tmp=200GB

#SBATCH --error=logs/ntu/CS/eval/error_cnn.txt
#SBATCH --output=logs/ntu/CS/eval/eval_cnn.txt

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

# Create directory on job file system to move dataset to
mkdir $JOBFS/data

# ---------------------------------------------------------
dataset="ntu"
model_type="cnn" # or base, abs...
evaluation="CS"
run_name="${dataset}_${evaluation}_${model_type}"
# ---------------------------------------------------------
mkdir -p ./results/${dataset}/${evaluation}/eval/ # eg. results/ntu/CS/eval


# Move the dataset (large) to the $JOBFS so it can be memory mapped during training
if [ "$model_type" = "base" ]; then
    cp ./data/ntu/aligned_data/NTU60_${evaluation}-pose_aligned.npz ${JOBFS}/data/
    echo "Copied pose_aligned.npz"
elif [ "$model_type" = "abs" ] || [ "$model_type" = "avg" ] || [ "$model_type" = "cnn" ]; then
    cp ./data/ntu/aligned_data/NTU60_${evaluation}-flowpose_aligned.npz ${JOBFS}/data/
    echo "Copied flowpose_aligned.npz"
fi

echo "Dataset copied to JOBFS: $(ls ${JOBFS}/data)"

# -d : config (eg. ntu)
# -m : model_type (base, cnn, abs, avg)
#         ./config/{dataset}/{model_type}.yaml
# -e : evaluation (eg. CV for ntu, CSet for ntu120, 2 for ucf101)
# -r : run_name (eg. ntu_CV_cnn)
#         checkpoint_file = '{arg.save_location}/{evaluation}/run_name'
# -s : save_name (eg. 'eval/run_name')
#         results_file = '{arg.save_location}/{evaluation}/save_name'
# -v : verbose

python training/eval_infogcn.py \
       -d "${dataset}" \
       -m $model_type \
       -e "${evaluation}" \
       -r "FULL_FLOW/nturgbd_CS_cnn" \
       -s ./results/${dataset}/${evaluation}/eval/${run_name}.pt \
       --data_path_overwrite "${JOBFS}/data" \
       # -v
