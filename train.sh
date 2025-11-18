#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

# Print the date...
date

# Create directory on job file system to move dataset to
mkdir $JOBFS/data

# Double check that from this script, ${copy_file} exists!
if [ ! -f $copy_file ]; then
    echo "File ${copy_file} not found!"
    exit 1
fi

# Move the dataset (large) to the $JOBFS so it can be memory mapped during training
echo "Copying file: ${copy_file} to ${JOBFS}/data"
cp ${copy_file} "${JOBFS}/data/"

# Echo which dataset we copied
echo "Dataset copied to JOBFS: $(ls ${JOBFS}/data)"

# Make sure the data path after copying makes sense...
filename=$(basename "$copy_file")
data_path="${JOBFS}/data/${filename}"
echo "Data path: ${data_path}"

# Echo the run name (have it at the top of the log file)
echo -e "Run name: ${run_name}\n"


# -m : model (eg. infogcn2, msg3d)
# -d : dataset (eg. ntu)
# -p : phase (for train.sh, it's train)
# -f : flow_embedding (base, cnn, abs, avg)
#         ./config/{dataset}/{flow_embedding}.yaml
# -e : evaluation (eg. CV for ntu, CSet for ntu120, 2 for ucf101)
# -o : observation ratio for training and testing (default 1.0)
# -r : run_name (eg. nturgbd_CV_cnn_full_flow)
#         checkpoint_file = '{arg.save_location}/{evaluation}/run_name'
# --desc : description (eg. 'nturgbd-cross-view cnn full flow')
# --data_path_overwrite : overwrites the path to the dataset
# -v : verbose

srun python main.py \
     -m "${model}" \
     -d "${dataset}" \
     -p "${phase}" \
     -f "${flow_embedding}" \
     -e "${evaluation}" \
     -o "${obs_ratio}" \
     -r "${run_name}" \
     --desc "${desc}" \
     --data_path_overwrite "${data_path}" \
     --debug "$debug" \
     -v
     # -v \
     # --debug
