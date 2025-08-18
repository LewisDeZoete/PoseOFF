#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

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



# -d : dataset (eg. ntu)
# -p : phase (for train.sh, it's train)
# -m : model_type (base, cnn, abs, avg)
#         ./config/{dataset}/{model_type}.yaml
# -e : evaluation (eg. CV for ntu, CSet for ntu120, 2 for ucf101)
# -r : run_name (eg. nturgbd_CV_cnn_full_flow)
#         checkpoint_file = '{arg.save_location}/{evaluation}/run_name'
# --desc : description (eg. 'nturgbd-cross-view cnn full flow')
# --data_path_overwrite : overwrites the path to the dataset
# -v : verbose

srun python main.py \
     -d "${dataset}" \
     -p "${phase}" \
     -m "${model_type}" \
     -e "${evaluation}" \
     -r "${run_name}" \
     --desc "${desc}" \
     --data_path_overwrite "${data_path}" \
     -v
