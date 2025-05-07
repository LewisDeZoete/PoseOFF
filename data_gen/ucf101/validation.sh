#!/bin/bash
#SBATCH --job-name=UCF-101_validation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --output='logs/EXTRACT/ucf101/UCF-101_validation.txt'
#SBATCH --error='logs/EXTRACT/ucf101/error_UCF-101_validation.txt'

# Activate the environment
source ../environment/bin/activate

# Run the validation script
python ./data_gen/ucf101/gendata_validation.py

# Check if the Python script ran successfully
if [ $? -eq 0 ]; then
  # Clean up
  rm -r ./TMP
  rm ./data_gen/ucf101/incomplete_classes.txt
else
  echo "Validation script encountered an error. Cleanup skipped."
fi