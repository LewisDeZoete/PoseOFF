#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00

# Activate the environment
source ../environment/bin/activate

# Run the validation script
python ./data_gen/ucf101/gendata_validation.py

# Check if the Python script ran successfully
if [ $? -eq 0 ]; then
  # Clean up
  rm -r ./TMP
  echo "Validation script completed"
else
  echo "Validation script encountered an error. Cleanup skipped."
fi
