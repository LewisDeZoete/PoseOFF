#!/bin/bash
# UCF-101 Extract Script
# This script is designed to extract features from the UCF-101 dataset using a specified modality.
# It creates the necessary annotations and incomplete classes files, and then submits a job array for processing.
# It also submits a validation job after the array job completes.

# ---------- GENERAL PROCESS: ----------
#     UCF-101_annotations.py
#     extract_utils.py
#     for class in incomplete_classes.txt
#         extract.sh
#     validation.sh
# TODO: align.sh
# --------------------------------------

# Usage: bash data_gen/ucf101/UCF-101_extract.sh


modality=flowpose

# Activate the environment
source ../environment/bin/activate

mkdir -p ./TMP
python ./data_gen/ucf101/UCF-101_annotations.py # creates the UCF-101_annotations.txt
python ./data_gen/utils/extract_utils.py -m $modality # creates the incomplete_classes.txt

# Read the number of incomplete classes
NUM_INCOMPLETE_CLASSES=$(wc -l < ./data_gen/ucf101/incomplete_classes.txt)
echo "Number of incomplete classes: $NUM_INCOMPLETE_CLASSES"

# Ensure NUM_INCOMPLETE_CLASSES is a valid integer
if ! [[ "$NUM_INCOMPLETE_CLASSES" =~ ^[0-9]+$ ]]; then
  echo "Error: NUM_INCOMPLETE_CLASSES is not a valid integer."
  exit 1
fi

# JOB ARRAY STARTS
# sbatch --array=0-$(($NUM_INCOMPLETE_CLASSES-1)) --export=modality=$modality ./data_gen/utils/extract.sh
array_job_id=$(sbatch --array=0-$(($NUM_INCOMPLETE_CLASSES-1)) --export=modality=$modality ./data_gen/ucf101/extractors/extract.sh | awk '{print $4}')

# # Submit the validation job with a dependency on the array job
sbatch --dependency=afterok:$array_job_id ./data_gen/ucf101/validation.sh
# validation_job_id=$(sbatch --dependency=afterok:$array_job_id ./data_gen/ucf101/validation.sh | awk '{print $4}')

# # Submit the next job with a dependency on the validation job
# sbatch --dependency=afterok:$validation_job_id ./data_gen/ucf101/align.sh