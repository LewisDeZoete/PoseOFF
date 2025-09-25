#!/usr/bin/env sh


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

export dilation=1
modalities=("flow" "pose")

# Create temporary directory to add video instances that don't contain poses
mkdir -p ./TMP

echo "Creating ucf101 annotations..."
python ./data_gen/ucf101/get_ucf101_annotations.py
echo -e "\tAnnotations created!"

# Array to store dependencies that must be met before flowpose extract is run...
flowpose_dependencies=()
declare -i NUM_INCOMPLETE_CLASSES


get_num_incomplete_classes () {
    # Create the ./data/ucf101/statistics/{modality}_incomplete_classes.yaml file
    python ./data_gen/utils/extract_utils.py -m $modality

    # Read the number of incomplete classes
    NUM_INCOMPLETE_CLASSES=$(grep -E '^[^[:space:]].*:$' \
        ./data/ucf101/statistics/${modality}_incomplete_classes.yaml | wc -l)

    # Ensure NUM_INCOMPLETE_CLASSES is a valid integer
    if ! [[ "$NUM_INCOMPLETE_CLASSES" =~ ^[0-9]+$ ]]; then
        exit 1
    fi
}


for modality in "${modalities[@]}"; do
    echo "MODALITY: $modality"
    export modality

    # Call function to find the number of incomplete classes
    get_num_incomplete_classes
    echo -e "\tNumber of incomplete classes for $modality: $NUM_INCOMPLETE_CLASSES"

    # Check if NUM_INCOMPLETE_CLASSES is zero...
    if [[ $NUM_INCOMPLETE_CLASSES = 0 ]]; then
        echo -e "\tNo classes to process for ${modality}"
        continue
    fi

    # JOB ARRAY STARTS
    extract_job_id=$(sbatch \
        --export=ALL \
        --job-name=ucf101_extract_${modality} \
        --array=0-$(($NUM_INCOMPLETE_CLASSES-1)) \
        --time=0:30:00 \
        --gres=gpu:1 \
        --parsable \
        --output=./logs/EXTRACT/ucf101/ucf101_extract_${modality}.out \
        --error=./logs/EXTRACT/ucf101/error_ucf101_extract_${modality}.out \
        ./data_gen/ucf101/extractors/extract.sh)
    echo "Submitted a batch of ${NUM_INCOMPLETE_CLASSES} jobs to extract ${modality}"

    # TODO: Run validation job before extract scripts? (it just removes empty annotations)
    # Submit the validation job with a dependency on the array job
    validation_job_id=$(sbatch \
        --export=ALL \
        --job-name=ucf101_validation_${modality} \
        --parsable \
        --output=./logs/EXTRACT/ucf101/ucf101_validation_${modality}.out \
        --error=./logs/EXTRACT/ucf101/error_ucf101_validation_${modality}.out \
        --dependency=afterok:$extract_job_id \
        ./data_gen/ucf101/extractors/validation.sh)
    echo "Submitted validation job dependent on successful extraction"

    flowpose_dependencies+=("${validation_job_id}")
done


# Now check if we need to run for the flowpose modality...
export modality="flowpose"
echo "MODALITY: $modality"
get_num_incomplete_classes
echo -e "\tNumber of incomplete classes for $modality: $NUM_INCOMPLETE_CLASSES"

# Check if NUM_INCOMPLETE_CLASSES is zero...
if ! [[ $NUM_INCOMPLETE_CLASSES = 0 ]]; then
    # If flowpose_dependencies is not empty, create a dependency argument string
    if [[ "${#flowpose_dependencies[@]}" -gt 0 ]]; then
        dep_arg="--dependency=afterok:$(IFS=:; echo "${flowpose_dependencies[*]}")"
    fi

    # JOB ARRAY STARTS
    extract_job_id=$(sbatch \
        --export=ALL \
        --job-name=ucf101_extract_flowpose \
        --array=0-$(($NUM_INCOMPLETE_CLASSES-1)) \
        --time=0:05:00 \
        --parsable \
        --output=./logs/EXTRACT/ucf101/ucf101_extract_flowpose_D${dilation}.out \
        --error=./logs/EXTRACT/ucf101/error_ucf101_extract_flowpose_D${dilation}.out \
        $dep_arg \
        ./data_gen/ucf101/extractors/extract.sh)
    echo "Submitted a batch of ${NUM_INCOMPLETE_CLASSES} jobs to extract flowpose"

    # Submit the validation job with a dependency on the array job
    validation_job_id=$(sbatch \
        --export=ALL \
        --job-name=ucf101_validation_flowpose \
        --parsable \
        --output=./logs/EXTRACT/ucf101/ucf101_validation_flowpose_D${dilation}.out \
        --error=./logs/EXTRACT/ucf101/error_ucf101_validation_flowpose_D${dilation}.out \
        --dependency=afterok:$extract_job_id \
        ./data_gen/ucf101/extractors/validation.sh)
    echo "Submitted validation job dependent on successful extraction"

fi


# Check if the aligned dataset exists...
aligned=true
aligned_datafolder="./data/ucf101/aligned_data"
mkdir -p ${aligned_datafolder}
for evaluation in 1 2 3; do
    file="$aligned_datafolder/ucf101_0${evaluation}_D${dilation}.npz"
    if [[ ! -f "$file" ]]; then
        aligned=false
        break
    fi
done

if ! $aligned; then
    echo "UCF-101 data needs realignment"
    # Submit the validation job with a dependency on the array job
    validation_job_id=$(sbatch \
        --export=ALL \
        --job-name=ucf101_validation_D${dilation} \
        --parsable \
        --output=./logs/EXTRACT/ucf101/ucf101_validation_flowpose_D${dilation}.out \
        --error=./logs/EXTRACT/ucf101/error_ucf101_validation_flowpose_D${dilation}.out \
        ./data_gen/ucf101/extractors/validation.sh)
    echo "Submitted validation job dependent on successful extraction"

    sbatch \
        --export=ALL \
        --job-name=ucf101_align_D${dilation} \
        --error=logs/EXTRACT/ucf101/error_ucf101_align_D${dilation}.out \
        --output=logs/EXTRACT/ucf101/ucf101_align_D${dilation}.out \
        --dependency=afterok:$validation_job_id \
        ./data_gen/ucf101/extractors/align.sh
fi
