#!/bin/bash
# Testing a unified file for sending of model training jobs...
# USAGE: Edit the variables below based on the
#        network you'd like to train or evaluate
# ```
# bash run.sh
# ```

export dataset="ntu120" # (ntu, ntu120, ucf101)
export phase="eval" # (train/eval)
export model_type="cnn" # (base, cnn, abs, avg, cnn_D3...)
export dilation=3
export modifier="D3" # still loads ${dataset}/{model_type}.yaml config... adds to run_name
export evaluation="CSet" # (CS/CV, CSub/CSet, 1/2/3)

export run_name="${dataset}_${evaluation}_${model_type}_${modifier}"
export desc="${phase} ${dataset} ${evaluation} ${model_type} re-running dilation 1 for testing..." # Can change this as you need!

# ---------------------------------------------------------
# Create the results and logs folders as needed...
mkdir -p ./results/${dataset}/${evaluation}/${phase}
mkdir -p ./logs/${dataset}/${evaluation}/${phase}
# e.g. ./results/ntu/CS/train/


case $dataset in
    # -------------------------------------------------------------
    # NTU RGB+D
    # -------------------------------------------------------------
    "ntu")
        if ! [[ $evaluation == "CS" || $evaluation == "CV" ]]; then
           echo "Wrong evaluation type for ntu: '${evaluation}', must be (CS/CV)"
           exit 1
        fi
        case $model_type in
            "base")
                export copy_file="./data/ntu/aligned_data/${dataset}_${evaluation}-pose_D${dilation}_aligned.npz"
                time=4:00:00
                mem=3GB
                tmp=15GB;;
            "abs" | "avg")
                export copy_file="./data/${dataset}/aligned_data/${dataset}_${evaluation}-flowpose_D${dilation}_aligned.npz"
                time=12:00:00
                mem=5GB
                tmp=200GB;;
            "cnn")
                export copy_file="./data/${dataset}/aligned_data/${dataset}_${evaluation}-flowpose_D${dilation}_aligned.npz"
                time=16:00:00
                mem=10GB
                tmp=200GB;;
            *)
                echo "Wrong model type '${model_type}', must be (base, cnn, abs, avg)"
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
        case $model_type in
            "base")
                export copy_file="./data/ntu120/aligned_data/ntu120_${evaluation}-pose_aligned.npz"
                time=7:00:00
                mem=5GB
                tmp=30GB;;
            "abs" | "avg")
                export copy_file="./data/ntu120/aligned_data/ntu120_${evaluation}-flowpose_D${dilation}_aligned.npz"
                time=21:00:00
                mem=7GB
                tmp=400GB;;
            "cnn")
                export copy_file="./data/ntu120/aligned_data/ntu120_${evaluation}-flowpose_D${dilation}_aligned.npz"
                time=24:00:00
                mem=15GB
                tmp=400GB;;
            *)
                echo "Wrong model type '${model_type}', must be (base, cnn, abs, avg)"
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
        time=2:00:00
        mem=5GB
        tmp=30GB
        # case $model_type in
        #     "base")
        #         time=7:00:00
        #         mem=5GB
        #         tmp=30GB;;
        #     "abs" | "avg")
        #         echo "not_base";;
        #     "cnn")
        #         echo "CNN";;
        #     *)
        #         echo "Wrong model type '${model_type}', must be (base, cnn, abs, avg)"
        #         exit 1
        # esac;;
esac


if [ $phase = "eval" ]; then
    # export run_name="EVAL-${dataset}_${evaluation}_${model_type}"
    mem=15GB
    time=0:40:00
fi

# Check if copy file exists
if [ ! -f $copy_file ]; then
    echo "File ${copy_file} not found!"
    exit 1
fi

# Check if run file already exists
if [ -f results/$dataset/$evaluation/train/$run_name.pt ]; then
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

error="logs/${dataset}/${evaluation}/${phase}/error_${model_type}_${modifier}.out"
output="logs/${dataset}/${evaluation}/${phase}/${phase}_${model_type}_${modifier}.out"

echo "${phase} ${dataset} ${evaluation} ${model_type} variables:"
echo -e "\terror: ${error}"
echo -e "\toutput: ${output}"
echo -e "\ttime: ${time}"
echo -e "\tmem: ${mem}"
echo -e "\ttmp: ${tmp}"
echo -e "\tcopy file: ${copy_file}"

echo -e "\tdataset: ${dataset}" # (ntu, ntu120, ucf101)
echo -e "\tmodel type: ${model_type}" # (base, cnn, abs, avg)
echo -e "\tdilation: ${dilation}"
echo -e "\tevaluation: ${evaluation}" # (CS/CV, CSub/CSet, 1/2/3)
echo -e "\trun name: ${run_name}"

# SBATCH using the variables set by this file
sbatch --export=ALL \
       --job-name=$run_name \
       --time=$time \
       --mem=$mem \
       --tmp=$tmp \
       --output=$output \
       --error=$error \
       ./train.sh
