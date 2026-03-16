#!/bin/bash

# ------------------------------------------
# USAGE: bash data_gen/ntu/ntu_gendata.sh

# Change the dataset to 'ntu' or 'ntu120'
# Change the dilation value (1 is default)

export dataset=ntu
export dilation=1
export flow_type=LK # LK, norm, RAFT, or WAFT

# Ensure the right flow type is being input...
_flow_types=("LK" "norm" "RAFT" "WAFT")
if [[ ${_flow_types[@]} =~ $flow_type ]]
then
    echo "Flow type: ${flow_type}"
else
    echo "Please edit the bash file, setting 'flow_type' to one of:"
    echo "'LK', 'norm', 'RAFT', or 'WAFT'"
    exit 1
fi


# --------------------------------------------

mkdir -p ./logs/EXTRACT/${dataset}
mkdir -p ./logs/EXTRACT/${dataset}/poseoff
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


# case $flow_type in
#     # -------------------------------------------------------------
#     # PoseOFF SAMPLING SPARSE (LK)
#     # -------------------------------------------------------------
#     "LK")
#         # Both datasets have ~57000 samples so it would need >28 batches of 2000 to process
#         flow_extract_id=$(sbatch --array=1-29 \
#             --export=ALL \
#             --job-name=${dataset}_poseoff_D${dilation} \
#             --time=02:00:00 \
#             --output=./logs/EXTRACT/${dataset}/poseoff/${dataset}_poseoff_%a.out \
#             --error=./logs/EXTRACT/${dataset}/poseoff/error_${dataset}_poseoff_%a.out \
#             ./data_gen/ntu/extractors/get_flows.sh | awk '{print $4}')
#         echo "Flow extract job id: $flow_extract_id";;

#     # -------------------------------------------------------------
#     # PoseOFF SAMPLING DENSE
#     # -------------------------------------------------------------
#     *)
#         # Both datasets have ~57000 samples so it would need >28 batches of 2000 to process
#         flow_extract_id=$(sbatch --array=1-29 \
#               --export=ALL \
#               --job-name=${dataset}_poseoff_D${dilation} \
#               --time=14:00:00 \
#               --gres=gpu:1 \
#               --output=./logs/EXTRACT/${dataset}/poseoff/${dataset}_poseoff_%a.out \
#               --error=./logs/EXTRACT/${dataset}/poseoff/error_${dataset}_poseoff_%a.out \
#               ./data_gen/ntu/extractors/get_flows.sh | awk '{print $4}')
#         echo "Flow extract job id: $flow_extract_id"
# esac

# # Concatenates all of the data under data/ntu*/flow_data/export_tmp
# # SAVES AS: data/ntu*/flow_data/flow_data_D{dilation}.pkl
# concat_id=$(sbatch \
#         --export=ALL \
#         --job-name=${dataset}_concat_poseoff_D${dilation} \
#         --output=./logs/EXTRACT/${dataset}/${dataset}_concat_poseoff.out \
#         --error=./logs/EXTRACT/${dataset}/error_${dataset}_concat_poseoff.out \
#         --dependency=afterok:$flow_extract_id \
#         ./data_gen/ntu/extractors/concat_poseoff.sh | awk '{print $4}')
# echo "Concat poseoff job id: $concat_id"

# ------------------------------------------
# ALIGN SEQUENCES
# ------------------------------------------
# align_id=$(sbatch \
#     --export=ALL \
#     --job-name=${dataset}_seq_transform_D${dilation} \
#     --output=./logs/EXTRACT/${dataset}/${dataset}_seq_transform.out \
#     --error=./logs/EXTRACT/${dataset}/error_${dataset}_seq_transform.out \
#     --dependency=afterok:$concat_id \
#     ./data_gen/ntu/extractors/seq_transformation.sh | awk '{print $4}')
# echo "Align sequence job id: $align_id"
align_id=$(sbatch \
    --export=ALL \
    --job-name=${dataset}_seq_transform_D${dilation} \
    --output=./logs/EXTRACT/${dataset}/${dataset}_seq_transform.out \
    --error=./logs/EXTRACT/${dataset}/error_${dataset}_seq_transform.out \
    ./data_gen/ntu/extractors/seq_transformation.sh | awk '{print $4}')
echo "Align sequence job id: $align_id"

# ------------------------------------------
# GET MEAN MAP
# ------------------------------------------
# sbatch --export=ALL \
#     --job-name=${dataset}_get_mean_map \
#     --output=./logs/EXTRACT/${dataset}/${dataset}_get_mean_map.out \
#     --error=./logs/EXTRACT/${dataset}/error_${dataset}_get_mean_map.out \
#     --dependency=afterok:$align_id \
#     ./data_gen/ntu/extractors/get_mean_map.sh

