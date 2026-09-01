#!/usr/bin/env sh


# UCF-101 Extract Script
# This script is designed to extract features from the UCF-101 dataset using a specified modality.
# It creates the necessary annotations and incomplete classes files, and then submits a job array for processing.
# It also submits a validation job after the array job completes.

# ---------- GENERAL PROCESS: ----------
# FIRST - if you're regenerating PoseOFF, change `redo_modalities=("poseoff")`
#      this runs: rm -rf ./data/ucf101/poseoff
#      (this is okay, final export is under ./data/ucf101/aligned_data)
#
#
# Usage: bash data_gen/ucf101/ucf101_gendata.sh
#
# HOW THE SCRIPT RUNS:
#     ucf101_annotations.py
#     extract_utils.py
#     for class in incomplete_classes.txt
#         extract.sh
#     validation.sh
#     align.sh
#
# Once extraction is complete, aligned data is written to `./data/ucf101/aligned_data`
# --------------------------------------

export dilation=3 # 1,2,3,...
export flow_type="NF" # RAFT, LK, NF, ...

# Which modalities to check incomplete classes for and run
modalities=("poseoff") # "pose", "flow", "poseoff"
# If passed, gives the option to delete the folder containing the data...
redo_modalities=("poseoff") # "pose", "flow", "poseoff"

# If redo_modalities is not empty, check if you want to delete the folders...
if ! [[ ${#redo_modalities[@]} == 0 ]]; then
    read -r -p "Do you want to delete the folders for: ./data/ucf101/${redo_modalities[@]}? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            for redo_modality in "${redo_modalities[@]}"; do
                echo "Deleting folder ./data/ucf101/${redo_modality}"
                rm -rf "./data/ucf101/${redo_modality}"
                echo "Deleted folder ./data/ucf101/${redo_modality}"
            done;;
        *)
            echo "No folders deleted..."
    esac
fi

# Create temporary directory to add video instances that don't contain poses
mkdir -p ./TMP

echo "Creating ucf101 annotations..."
python ./data_gen/ucf101/get_ucf101_annotations.py
echo -e "\tAnnotations created!"

# Array to store dependencies that must be met before poseoff extract is run...
poseoff_dependencies=()
declare -i NUM_INCOMPLETE_CLASSES


get_num_incomplete_classes () {
    # Create the ./data/ucf101/statistics/{modality}_incomplete_classes.yaml file
    # Also ensures the modality folder is created...
    python ./data_gen/utils/extract_utils.py -m $modality

    # Read the number of incomplete classes
    NUM_INCOMPLETE_CLASSES=$(grep -E '^[^[:space:]].*:$' \
        ./data/ucf101/statistics/${modality}_incomplete_classes.yaml | wc -l)

    # Ensure NUM_INCOMPLETE_CLASSES is a valid integer
    if ! [[ "$NUM_INCOMPLETE_CLASSES" =~ ^[0-9]+$ ]]; then
        echo "./data_gen/utils/extract_utils.py -m ${modality} failed to return a valid int..."
        exit 1
    fi
}


for modality in "${modalities[@]}"; do
    # Skip poseoff for now, this is done to ensure that pose and flow are extracted first...
    if [[ "$modality" == "poseoff" ]]; then
        continue
    fi

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
        --output=./logs/EXTRACT/ucf101/ucf101_extract_${modality}_D${dilation}.out \
        --error=./logs/EXTRACT/ucf101/error_ucf101_extract_${modality}_D${dilation}.out \
        ./data_gen/ucf101/extractors/extract.sh)
    echo "Submitted a batch of ${NUM_INCOMPLETE_CLASSES} jobs to extract ${modality} (JOBID: ${extract_job_id})"

    # Submit the validation job with a dependency on the array job
    validation_job_id=$(sbatch \
        --export=ALL \
        --job-name=ucf101_validation_${modality} \
        --parsable \
        --output=./logs/EXTRACT/ucf101/ucf101_validation_${modality}_D${dilation}.out \
        --error=./logs/EXTRACT/ucf101/error_ucf101_validation_${modality}_D${dilation}.out \
        --dependency=afterok:$extract_job_id \
        ./data_gen/ucf101/extractors/validation.sh)
    echo "Submitted validation job dependent on successful extraction (JOBID: ${validation_job_id})"

    poseoff_dependencies+=("${validation_job_id}")
    echo $poseoff_dependencies
done


# Now check if we need to run for the poseoff modality...
export modality="poseoff"
echo "MODALITY: $modality"
get_num_incomplete_classes
echo -e "\tNumber of incomplete classes for $modality: $NUM_INCOMPLETE_CLASSES"

# Check if NUM_INCOMPLETE_CLASSES is zero...
if ! [[ $NUM_INCOMPLETE_CLASSES = 0 ]]; then
    # If poseoff_dependencies is not empty, create a dependency argument string
    if [[ "${#poseoff_dependencies[@]}" -gt 0 ]]; then
        dep_arg="--dependency=afterok:$(IFS=:; echo "${poseoff_dependencies[*]}")"
    fi

    # JOB ARRAY STARTS
    poseoff_extract_job_id=$(sbatch \
        --export=ALL \
        --job-name=ucf101_extract_poseoff \
        --array=0-$(($NUM_INCOMPLETE_CLASSES-1)) \
        --time=0:15:00 \
        --parsable \
        --output=./logs/EXTRACT/ucf101/ucf101_extract_poseoff_${flow_type}_D${dilation}.out \
        --error=./logs/EXTRACT/ucf101/error_ucf101_extract_poseoff_${flow_type}_D${dilation}.out \
        $dep_arg \
        ./data_gen/ucf101/extractors/extract.sh)
    echo "Submitted a batch of ${NUM_INCOMPLETE_CLASSES} jobs to extract poseoff (JOBID: ${poseoff_extract_job_id})"

    # Submit the validation job with a dependency on the array job
    validation_job_id=$(sbatch \
        --export=ALL \
        --job-name=ucf101_validation_poseoff \
        --parsable \
        --output=./logs/EXTRACT/ucf101/ucf101_validation_poseoff_${flow_type}_D${dilation}.out \
        --error=./logs/EXTRACT/ucf101/error_ucf101_validation_poseoff_${flow_type}_D${dilation}.out \
        --dependency=afterok:$poseoff_extract_job_id \
        ./data_gen/ucf101/extractors/validation.sh)
    echo "Submitted validation job dependent on successful extraction (JOBID: ${validation_job_id})"
fi


# Check if the aligned dataset exists...
aligned=true
aligned_datafolder="./data/ucf101/aligned_data/${flow_type}"
mkdir -p ${aligned_datafolder}
for evaluation in 1 2 3; do
    file="$aligned_datafolder/ucf101_0${evaluation}-poseoff_${flow_type}_D${dilation}.npz"
    if [[ ! -f "$file" ]]; then
        aligned=false
        break
    fi
done

if ! $aligned; then
    # Create a job depencency on whether the extract being complete
    if ! [[ $NUM_INCOMPLETE_CLASSES = 0 ]]; then
        dep_arg="--dependency=afterok:$(IF=:; echo "${validation_job_id}")"
    fi
    echo "UCF-101 data needs realignment"
    # Submit the validation job with a dependency on the array job
    validation_job_id=$(sbatch \
        --export=ALL \
        --job-name=ucf101_validation_D${dilation} \
        --parsable \
        --output=./logs/EXTRACT/ucf101/ucf101_validation_poseoff_${flow_type}_D${dilation}.out \
        --error=./logs/EXTRACT/ucf101/error_ucf101_validation_poseoff_${flow_type}_D${dilation}.out \
        $dep_arg \
        ./data_gen/ucf101/extractors/validation.sh)
    echo "Submitted validation job ${validation_job_id}"

    sbatch \
        --export=ALL \
        --job-name=ucf101_align_D${dilation} \
        --error=logs/EXTRACT/ucf101/error_ucf101_align_${flow_type}_D${dilation}.out \
        --output=logs/EXTRACT/ucf101/ucf101_align_${flow_type}_D${dilation}.out \
        --dependency=afterok:$validation_job_id \
        ./data_gen/ucf101/extractors/align.sh
fi
