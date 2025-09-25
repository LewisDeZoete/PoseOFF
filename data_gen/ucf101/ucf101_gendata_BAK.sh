#!/bin/bash
# UCF-101 Extract Script
# This script is designed to extract features from the UCF-101 dataset using a specified modality.
# It creates the necessary annotations and incomplete classes files, and then submits a job array for processing.
# It also submits a validation job after the array job completes.

# ---------- GENERAL PROCESS: ----------
# Usage: bash data_gen/ucf101/ucf101_gendata.sh
#
# HOW THE SCRIPT RUNS:
#     ucf101_annotations.py
#     extract_utils.py
#     for class in incomplete_classes.txt
#         extract.sh
#     validation.sh
# TODO: align.sh
# --------------------------------------

modality="flowpose"

# Activate the environment
source ../environment/bin/activate

mkdir -p ./TMP
echo "Creating ucf101 annotations..."
# creates ./data/ucf101/ucf101_annotations.txt
python ./data_gen/ucf101/get_ucf101_annotations.py
echo -e "\tAnnotations created!"
echo "Determining which classes are incomplete..."
# Create the ./data/ucf101/statistics/{modality}_incomplete_classes.yaml file
python ./data_gen/utils/extract_utils.py -m $modality

# Read the number of incomplete classes
NUM_INCOMPLETE_CLASSES=$(grep -E '^[^[:space:]].*:$' \
    ./data/ucf101/statistics/${modality}_incomplete_classes.yaml | wc -l)
echo "Number of incomplete classes: $NUM_INCOMPLETE_CLASSES"

# Ensure NUM_INCOMPLETE_CLASSES is a valid integer
if ! [[ "$NUM_INCOMPLETE_CLASSES" =~ ^[0-9]+$ ]]; then
  echo "Error: NUM_INCOMPLETE_CLASSES is not a valid integer."
  exit 1
fi
if [[ $NUM_INCOMPLETE_CLASSES = 0 ]]; then
    echo "No classes to process!"
    # Check if sequence re-alignment necessary...
    if [[ $align = true ]]; then
        exit 0
    fi
    exit 1
fi

# Depending on the modality, resource requirements are different
case $modality in
    "pose")
        time=0:30:00
        gres=gpu:1;;
    "flow")
        time=0:15:00
        gres=gpu:1;;
    "flowpose")
        time=0:05:00
        gres=gpu:0;;
esac


# JOB ARRAY STARTS
extract_job_id=$(sbatch \
    --job-name=ucf101_extract_${modality} \
    --array=0-$(($NUM_INCOMPLETE_CLASSES-1)) \
    --export=modality=$modality \
    --time=$time \
    --gres=$gres \
    --output=./logs/EXTRACT/ucf101/ucf101_extract_${modality}.out \
    --error=./logs/EXTRACT/ucf101/error_ucf101_extract_${modality}.out \
    ./data_gen/ucf101/extractors/extract.sh | awk '{print $4}')
echo "Submitted a batch of ${NUM_INCOMPLETE_CLASSES} jobs to extract ${modality}"

# # Submit the validation job with a dependency on the array job
# sbatch --dependency=afterok:$array_job_id ./data_gen/ucf101/validation.sh
validation_job_id=$(sbatch \
    --export=modality=$modality \
    --job-name=ucf101_validation_${modality} \
    --output=./logs/EXTRACT/ucf101/ucf101_validation_${modality}.out \
    --error=./logs/EXTRACT/ucf101/error_ucf101_validation_${modality}.out \
    --dependency=afterok:$extract_job_id \
    ./data_gen/ucf101/extractors/validation.sh | awk '{print $4}')
echo "Submitted validation job dependent on successful extraction"

# # # Submit the next job with a dependency on the validation job
# sbatch --dependency=afterok:$validation_job_id ./data_gen/ucf101/extractors/align.sh
