#!/bin/bash
# Testing a unified file for sending of model training jobs...

export dataset="ntu120" # (ntu, ntu120, ucf101)
export phase="train" # (train/eval)
export model_type="cnn" # (base, cnn, abs, avg)
export evaluation="CSub" # (CS/CV, CSub/CSet, 1/2/3)
export run_name="${dataset}_${evaluation}_${model_type}"
export desc="${phase} ${dataset} {evaluation} {model_type}" # Can change this as you need!

# ---------------------------------------------------------
# Create the results folder as needed...
mkdir -p ./results/${dataset}/${evaluation}/${phase}
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
                export copy_file="./data/ntu/aligned_data/NTU60_${evaluation}-pose_aligned.npz"
                time=4:00:00
                mem=3GB
                tmp=15GB;;
            "abs" | "avg")
                export copy_file="./data/ntu/aligned_data/NTU60_${evaluation}-flowpose_aligned.npz"
                time=12:00:00
                mem=3GB
                tmp=200GB;;
            "cnn")
                export copy_file="./data/ntu/aligned_data/NTU60_${evaluation}-flowpose_aligned.npz"
                time=13:00:00
                mem=10GB
                tmp=200GB;;
            *)
                echo "Wrong model type '${model_type}', must be (base, cnn, abs, avg)"
                exit 1
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
2                export copy_file="./data/ntu120/aligned_data/NTU120_${evaluation}-pose_aligned.npz"
                time=7:00:00
                mem=5GB
                tmp=30GB;;
            "abs" | "avg")
                export copy_file="./data/ntu120/aligned_data/NTU120_${evaluation}-flowpose_aligned.npz"
                time=18:30:00
                mem=5GB
                tmp=400GB;;
            "cnn")
                export copy_file="./data/ntu120/aligned_data/NTU120_${evaluation}-flowpose_aligned.npz"
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
        case $model_type in
            "base")
                export copy_file="./data/ucf101/aligned_data/NTU120_${evaluation}-flowpose_aligned.npz";;
            "abs" | "avg")
                echo "not_base";;
            "cnn")
                pass
        esac;;
esac

error="logs/${dataset}/${evaluation}/${phase}/error_${model_type}.out"
output="logs/${dataset}/${evaluation}/${phase}/train_${model_type}.out"

echo "${phase} ${dataset} ${evaluation} ${model_type} variables:"
echo -e "\terror: ${error}"
echo -e "\toutput: ${output}"
echo -e "\ttime: ${time}"
echo -e "\tmem: ${mem}"
echo -e "\ttmp: ${tmp}"
echo -e "\tcopy file: ${copy_file}"

echo -e "\tdataset: ${dataset}" # (ntu, ntu120, ucf101)
echo -e "\tmodel type: ${model_type}" # (base, cnn, abs, avg)
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
