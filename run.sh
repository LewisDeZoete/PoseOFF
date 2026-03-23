#!/bin/bash
# Testing a unified file for sending of model training jobs...
# USAGE: Edit the variables below based on the
#        network you'd like to train or evaluate
# ```
# bash run.sh
# ```

export model="stgcn2" # (infogcn2, msg3d, stgcn2)
export dataset="ntu" # (ntu, ntu120, ucf101)
export phase="train" # (train/eval)
export flow_embedding="cnn" # (base, cnn, abs, avg)
export flow_type="LK" # (RAFT, LK, norm, ???)
export dilation=4
export evaluation="CS" # (CS/CV, CSub/CSet, 1/2/3)
export obs_ratio="1.0"
export debug=false

# ALWAYS START MODIFIER WITH "_"
# still loads ${dataset}/{flow_embedding}.yaml config... adds to run_name
export modifier=""
if ! [[ $flow_embedding == "base" ]]; then
    export modifier="_${flow_type}_D${dilation}"
fi
# export modifier="_BLANK_EVAL" # Only used for timing model inference...

export run_name="${model}_${dataset}_${evaluation}_${flow_embedding}${modifier}" # save/load name
export job_name="${phase}-${run_name}"
export desc="${model}_${phase} ${dataset} ${evaluation} ${flow_embedding} observation ratio ${obs_ratio}" # Can change this as you need!


# ---------------------------------------------------------
# Create the results and logs folders as needed... e.g.
mkdir -p ./results/${model}/${dataset}/${evaluation}/${phase}
mkdir -p ./logs/${model}/${dataset}/${evaluation}/${phase}
# e.g. ./results/infogcn2/ntu/CS/train/

case $dataset in
    # -------------------------------------------------------------
    # NTU RGB+D
    # -------------------------------------------------------------
    "ntu")
        if ! [[ $evaluation == "CS" || $evaluation == "CV" ]]; then
           echo "Wrong evaluation type for ntu: '${evaluation}', must be (CS/CV)"
           exit 1
        fi
        case $flow_embedding in
            "base")
                export copy_file="./data/ntu/aligned_data/pose/ntu_${evaluation}-pose_aligned.npz"
                if [ $model == "msg3d" ]; then
                    time=10:00:00
                    mem=35GB
                else
                    time=4:00:00
                    mem=15GB
                fi
                tmp=15GB;;
            "abs" | "avg")
                export copy_file="./data/ntu/aligned_data/poseoff/${flow_type}/ntu_${evaluation}-poseoff_${flow_type}_D${dilation}_aligned.npz"
                if [ $model == "msg3d" ]; then
                    time=18:00:00
                else
                    time=12:00:00
                fi
                time=12:00:00
                mem=25GB
                tmp=200GB;;
            "cnn")
                export copy_file="./data/ntu/aligned_data/poseoff/${flow_type}/ntu_${evaluation}-poseoff_${flow_type}_D${dilation}_aligned.npz"
                if [ $model == "msg3d" ]; then
                    time=20:00:00
                else
                    time=16:00:00
                fi
                time=16:00:00
                mem=20GB
                tmp=200GB;;
            *)
                echo "Wrong model type '${flow_embedding}', must be (base, cnn, abs, avg)"
                echo "If you want to continue please change arguments in the file"
                read -p "Time : " time
                read -p "Memory : " mem
                read -p "Temporary storage : " tmp
                read -p "Copy filename: " copy_file
                export copy_file
        esac;;

    # -------------------------------------------------------------
    # NTU RGB+D 120
    # -------------------------------------------------------------    
    "ntu120")
        if ! [[ $evaluation == "CSub" || $evaluation == "CSet" ]]; then
           echo "Wrong evaluation type for ntu120: '${evaluation}', must be (CSub/CSet)"
           exit 1
        fi
        case $flow_embedding in
            "base")
                export copy_file="./data/ntu120/aligned_data/pose/ntu120_${evaluation}-pose_aligned.npz"
                if [ $model == "msg3d" ]; then
                    time=15:00:00
                    mem=50GB
                else
                    time=7:00:00
                    mem=10GB
                fi
                tmp=30GB
                if [[ $evaluation == "CSet" ]]; then
                    export copy_file="./data/ntu120/aligned_data/poseoff/${flow_type}/ntu120_${evaluation}-poseoff_${flow_type}_D${dilation}_aligned.npz"
                    time=15:00:00
                    tmp=400GB
                fi
                ;;
            "abs" | "avg")
                export copy_file="./data/ntu120/aligned_data/${flow_type}/ntu120_${evaluation}-poseoff_${flow_type}_D${dilation}_aligned.npz"
                time=21:00:00
                mem=17GB
                tmp=400GB;;
            "cnn")
                export copy_file="./data/ntu120/aligned_data/${flow_type}/ntu120_${evaluation}-poseoff_${flow_type}_D${dilation}_aligned.npz"
                time=24:00:00
                mem=35GB
                tmp=400GB;;
            *)
                echo "Wrong model type '${flow_embedding}', must be (base, cnn, abs, avg)"
                exit 1
        esac;;

    # -------------------------------------------------------------
    # UCF101
    # -------------------------------------------------------------
    "ucf101")
        if ! [[ $evaluation == "1" || $evaluation == "2" || $evaluation == "3" ]]; then
           echo "Wrong evaluation type for ucf101: '${evaluation}', must be (1/2/3)"
           exit 1
        fi
        export copy_file="./data/ucf101/aligned_data/ucf101_0${evaluation}_D${dilation}.npz"
        time=3:00:00
        mem=10GB
        tmp=30GB
        # case $flow_embedding in
        #     "base")
        #         time=7:00:00
        #         mem=5GB
        #         tmp=30GB;;
        #     "abs" | "avg")
        #         echo "not_base";;
        #     "cnn")
        #         echo "CNN";;
        #     *)
        #         echo "Wrong model type '${flow_embedding}', must be (base, cnn, abs, avg)"
        #         exit 1
        # esac;;
esac


# Check if results file exists (must exist for eval, check if you want to continue for train)
if [ $phase = "eval" ]; then
    mem=15GB
    time=0:30:00
    if ! [[ -f results/$model/$dataset/$evaluation/train/$run_name.pt ]]; then
        echo -e "Run ${run_name} does not exist..."
        read -r -p "Do you want to evaluate a randomised model? [y/N] " response
        case "$response" in
            [yY][eE][sS]|[yY])
                echo -e "\tContinuing eval of fresh model ${run_name}";;
            *)
                echo "Cancelling job batching..."
                exit 1;;
        esac
    fi
    error="logs/${model}/${dataset}/${evaluation}/${phase}/error_${flow_embedding}${modifier}_obs${obs_ratio}.out"
    output="logs/${model}/${dataset}/${evaluation}/${phase}/${phase}_${flow_embedding}${modifier}_obs${obs_ratio}.out"
    job_name="${job_name}_obs${obs_ratio}"
elif [ $phase == "train" ]; then
    if [[ -f results/$model/$dataset/$evaluation/train/$run_name.pt ]]; then
        echo -e "RUN ALREADY EXISTS: ${run_name}"
        read -r -p "Do you want to continue training? [y/N] " response
        case "$response" in
            [yY][eE][sS]|[yY])
                echo -e "\tContinuing run ${run_name}";;
            *)
                echo "Cancelling job batching..."
                exit 1;;
        esac
    fi
    error="logs/${model}/${dataset}/${evaluation}/${phase}/error_${flow_embedding}${modifier}.out"
    output="logs/${model}/${dataset}/${evaluation}/${phase}/${phase}_${flow_embedding}${modifier}.out"
fi

# Check if copy file exists
if [ ! -f $copy_file ]; then
    echo "File ${copy_file} not found!"
    exit 1
fi


echo "${phase} ${model} ${dataset} ${evaluation} ${flow_embedding} ${flow_type} variables:"
echo -e "\tmodel: ${model}"
echo -e "\terror: ${error}"
echo -e "\toutput: ${output}"
echo -e "\ttime: ${time}"
echo -e "\tmem: ${mem}"
echo -e "\ttmp: ${tmp}"
echo -e "\tcopy file: ${copy_file}"

echo -e "\tdataset: ${dataset}" # (ntu, ntu120, ucf101)
echo -e "\tmodel type: ${flow_embedding}" # (base, cnn, abs, avg)
echo -e "\tflow type: ${flow_type}" # (RAFT, LK, norm...)
echo -e "\tdilation: ${dilation}"
echo -e "\tevaluation: ${evaluation}" # (CS/CV, CSub/CSet, 1/2/3)
echo -e "\tobservation ratio: ${obs_ratio}"
echo -e "\tdebug: ${debug}"
echo -e "\trun name: ${run_name}"
echo -e "\tjob name: ${job_name}"

# SBATCH using the variables set by this file
sbatch --export=ALL \
       --job-name=$job_name \
       --time=$time \
       --mem=$mem \
       --tmp=$tmp \
       --output=$output \
       --error=$error \
       ./train.sh


# python main.py \
#     -m "${model}" \
#     -d "${dataset}" \
#     -p "${phase}" \
#     -f "${flow_embedding}" \
#     -e "${evaluation}" \
#     -o "${obs_ratio}" \
#     -r "${run_name}" \
#     --desc "${desc}" \
#     --data_path_overwrite "${data_path}" \
#     -v
