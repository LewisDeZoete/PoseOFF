#!/bin/bash

# Submit the job array and capture the job ID
JOBID=$(sbatch --array=0-100 --wrap="srun your_srun_command_here" | awk '{print $4}')

# Check if the job array submission was successful
if [[ -n "$JOBID" ]]; then
  # Submit the follow-up dependent job
  sbatch --dependency=afterok:${JOBID} --wrap="srun follow_up_command_here"
else
  echo "Job array submission failed."
fi