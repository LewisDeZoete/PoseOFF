#!/bin/bash

# ------------------------------------------
# USAGE: bash data_gen/NTU/ntu_gendata.sh

# Uncomment the blocks you want to run
# Change the dataset to 'ntu' or 'ntu120'

dataset=ntu
# ------------------------------------------

mkdir -p ./logs/EXTRACT/${dataset}
mkdir -p ./logs/EXTRACT/${dataset}/flowpose
mkdir -p ./data/${dataset}/flow_data/export_tmp

# # ------------------------------------------
# # GET RAW SKELETONS
# # ------------------------------------------
# sbatch --export=dataset=$dataset \
#        --job-name=${dataset}_get_raw_skes \
#        --output=./logs/EXTRACT/${dataset}/${dataset}_get_raw_skes.out \
#        --error=./logs/EXTRACT/${dataset}/errors_${dataset}_get_raw_skes.out \
#        ./data_gen/NTU/extractors/get_raw.sh


# # ------------------------------------------
# # DENOISE
# # ------------------------------------------
# sbatch --export=dataset=$dataset \
#        --job-name=${dataset}_denoise \
#        --output=./logs/EXTRACT/${dataset}/${dataset}_get_denoised.out \
#        --error=./logs/EXTRACT/${dataset}/errors_${dataset}_get_denoised.out \
#        ./data_gen/NTU/extractors/get_denoised.sh


# # ------------------------------------------
# # FLOWPOSE SAMPLING
# # ------------------------------------------
# flow_extract_id=$(sbatch --array=1-29 \
#       --export=dataset=$dataset \
#       --job-name=${dataset}_flowpose \
#       --output=./logs/EXTRACT/${dataset}/flowpose/${dataset}_flowpose_%a.out \
#       --error=./logs/EXTRACT/${dataset}/flowpose/error_${dataset}_flowpose_%a.out \
#       ./data_gen/NTU/extractors/get_flowpose.sh | awk '{print $4}')

# sbatch --dependency=afterok:$flow_extract_id \
# # sbatch \
#       --export=dataset=$dataset \
#       --output=./logs/EXTRACT/${dataset}/${dataset}_concat_flowpose.out \
#       --error=./logs/EXTRACT/${dataset}/error_${dataset}_concat_flowpose.out \
#       ./data_gen/NTU/extractors/concat_flowpose.sh


# ------------------------------------------
# ALIGN SEQUENCES
# ------------------------------------------
sbatch --export=dataset=$dataset \
       --job-name=${dataset}_seq_transform.txt \
       --output=./logs/EXTRACT/${dataset}/${dataset}_seq_transform.out \
       --error=./logs/EXTRACT/${dataset}/error_${dataset}_seq_transform.out \
       ./data_gen/NTU/extractors/seq_transformation.sh
