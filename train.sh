#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

export PYTHONUNBUFFERED=TRUE

source ../environment/bin/activate

# Create directory on job file system to move dataset to
mkdir $JOBFS/data

# Move the dataset (large) to the $JOBFS so it can be memory mapped during training
cp "${copy_file}" "${JOBFS}/data/"

echo "Dataset copied to JOBFS: $(ls ${JOBFS}/data)"


# -d : dataset (eg. ntu)
# -m : model_type (base, cnn, abs, avg)
#         ./config/{dataset}/{model_type}.yaml
# -e : evaluation (eg. CV for ntu, CSet for ntu120, 2 for ucf101)
# -r : run_name (eg. nturgbd_CV_cnn_full_flow)
#         checkpoint_file = '{arg.save_location}/{evaluation}/run_name'
# -d : description (eg. 'nturgbd-cross-view cnn full flow')
# -v : verbose

srun python main.py \
     -d "${dataset}" \
     -m "${model_type}" \
     -e "${evaluation}" \
     -r "${run_name}" \
     --desc "${desc}" \
     --data_path_overwrite "${JOBFS}/data" \
     -v
