#!/bin/bash

# ------------------------------------------
# USAGE: bash data_gen/ntu/ntu_gendata.sh

# Uncomment the blocks you want to run
# Change the dataset to 'ntu' or 'ntu120'
# Change the dilation value (1 is default)

export dataset=ntu120
export dilation=1
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
#        ./data_gen/ntu/extractors/get_raw.sh


# # ------------------------------------------
# # DENOISE
# # ------------------------------------------
# sbatch --export=dataset=$dataset \
#        --job-name=${dataset}_denoise \
#        --output=./logs/EXTRACT/${dataset}/${dataset}_get_denoised.out \
#        --error=./logs/EXTRACT/${dataset}/errors_${dataset}_get_denoised.out \
#        ./data_gen/ntu/extractors/get_denoised.sh


# ------------------------------------------
# FLOWPOSE SAMPLING
# ------------------------------------------
# Both datasets have ~57000 samples so it would need >28 batches of 2000 to process
flow_extract_id=$(sbatch --array=1-29 \
      --export=ALL \
      --job-name=${dataset}_flowpose_D${dilation} \
      --output=./logs/EXTRACT/${dataset}/flowpose/${dataset}_flowpose_%a.out \
      --error=./logs/EXTRACT/${dataset}/flowpose/error_${dataset}_flowpose_%a.out \
      ./data_gen/ntu/extractors/get_flowpose.sh | awk '{print $4}')
echo "Flow extract job id: $flow_extract_id"

# Concatenates all of the data under data/ntu*/flow_data/export_tmp
# SAVES AS: data/ntu*/flow_data/flow_data_D{dilation}.pkl
concat_id=$(sbatch \
        --export=ALL \
        --job-name=${dataset}_concat_flowpose_D${dilation} \
        --output=./logs/EXTRACT/${dataset}/${dataset}_concat_flowpose.out \
        --error=./logs/EXTRACT/${dataset}/error_${dataset}_concat_flowpose.out \
        --dependency=afterok:$flow_extract_id \
        ./data_gen/ntu/extractors/concat_flowpose.sh | awk '{print $4}')
echo "Concat flowpose job id: $concat_id"

# ------------------------------------------
# ALIGN SEQUENCES
# ------------------------------------------
sbatch --export=ALL \
       --job-name=${dataset}_seq_transform_D${dilation} \
       --output=./logs/EXTRACT/${dataset}/${dataset}_seq_transform.out \
       --error=./logs/EXTRACT/${dataset}/error_${dataset}_seq_transform.out \
       --dependency=afterok:$concat_id \
       ./data_gen/ntu/extractors/seq_transformation.sh
