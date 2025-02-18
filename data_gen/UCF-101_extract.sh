modality=flowpose

mkdir -p ./TMP
python ./data_gen/UCF-101_annotations.py
python ./data_gen/utils/extract_utils.py -m $modality # creates the incomplete_classes.txt

# Read the number of incomplete classes
NUM_INCOMPLETE_CLASSES=$(cat ./data_gen/num_incomplete.txt)
echo "Number of incomplete classes: $NUM_INCOMPLETE_CLASSES"

# Ensure NUM_INCOMPLETE_CLASSES is a valid integer
if ! [[ "$NUM_INCOMPLETE_CLASSES" =~ ^[0-9]+$ ]]; then
  echo "Error: NUM_INCOMPLETE_CLASSES is not a valid integer."
  exit 1
fi

# JOB ARRAY STARTS
# sbatch --array=0-$(($NUM_INCOMPLETE_CLASSES-1)) --export=modality=$modality ./data_gen/utils/extract.sh
array_job_id=$(sbatch --array=0-$(($NUM_INCOMPLETE_CLASSES-1)) --export=modality=$modality ./data_gen/utils/extract.sh | awk '{print $4}')

# Submit the validation job with a dependency on the array job
sbatch --dependency=afterok:$array_job_id ./data_gen/utils/validation.sh