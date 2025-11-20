#!/bin/bash
# Testing a unified file for sending of model training jobs...
# USAGE: Edit the variables below based on the
#        network you'd like to train or evaluate
# ```
# bash run.sh
# ```

export model="msg3d" # (infogcn2, msg3d, stgcn2)
export dataset="ntu" # (ntu, ntu120, ucf101)
export phase="train" # (train/eval)
export flow_embedding="base" # (base, cnn, abs, avg)
export dilation=3
export evaluation="CV" # (CS/CV, CSub/CSet, 1/2/3)
export obs_ratio="1.0"
export debug=false

export modifier="LR_fix" # still loads ${dataset}/{flow_embedding}.yaml config... adds to run_name

export run_name="${model}_${dataset}_${evaluation}_${flow_embedding}_${modifier}"
# export desc="${model}_${phase} ${dataset} ${evaluation} ${flow_embedding} observation ratio ${obs_ratio}, retraining model with mean and std norm" # Can change this as you need!
# export desc="${model}_${phase} ${dataset} ${evaluation} ${flow_embedding} Fixed LR Scheduler, original model with no data augmentation using the REPLAY padding method" # Can change this as you need!
export desc="${model}_${phase} ${dataset} ${evaluation} ${flow_embedding} Fixed LR Scheduler, STGC layers=2 no data augmentation using the REPLAY padding method" # Can change this as you need!


# ---------------------------------------------------------
# Create the results and logs folders as needed... e.g.
mkdir -p ./results/${model}/${dataset}/${evaluation}/${phase}
mkdir -p ./logs/${model}/${dataset}/${evaluation}/${phase}
# e.g. ./results/infogcn2/ntu/CS/train/

# # Base does not need a modifier at the end...
# if [ $flow_embedding == "base" ]; then
#     export run_name="${model}_${dataset}_${evaluation}_${flow_embedding}"
# fi

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
                export copy_file="./data/ntu/aligned_data/${dataset}_${evaluation}-pose_aligned_mean.npz"
                if [ $model == "msg3d" ]; then
                    time=10:00:00
                    mem=35GB
                else
                    time=4:00:00
                    mem=15GB
                fi
                tmp=15GB;;
            "abs" | "avg")
                export copy_file="./data/${dataset}/aligned_data/${dataset}_${evaluation}-flowpose_D${dilation}_aligned_mean.npz"
                if [ $model == "msg3d" ]; then
                    time=18:00:00
                else
                    time=12:00:00
                fi
                time=12:00:00
                mem=25GB
                tmp=200GB;;
            "cnn")
                export copy_file="./data/${dataset}/aligned_data/${dataset}_${evaluation}-flowpose_D${dilation}_aligned_mean.npz"
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
                export copy_file="./data/ntu120/aligned_data/ntu120_${evaluation}-pose_aligned_mean.npz"
                time=7:00:00
                mem=15GB
                tmp=30GB
                if [[ $evaluation == "CSet" ]]; then
                    export copy_file="./data/ntu120/aligned_data/ntu120_${evaluation}-flowpose_D3_aligned_mean.npz"
                    time=15:00:00
                    tmp=400GB
                fi
                ;;
            "abs" | "avg")
                export copy_file="./data/ntu120/aligned_data/ntu120_${evaluation}-flowpose_D${dilation}_aligned_mean.npz"
                time=21:00:00
                mem=17GB
                tmp=400GB;;
            "cnn")
                export copy_file="./data/ntu120/aligned_data/ntu120_${evaluation}-flowpose_D${dilation}_aligned_mean.npz"
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
        time=2:00:00
        mem=5GB
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
    time=0:10:00
    if ! [[ -f results/$model/$dataset/$evaluation/train/$run_name.pt ]]; then
        echo -e "Run ${run_name} does not exist..."
        exit 1;
    fi
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
fi

# Check if copy file exists
if [ ! -f $copy_file ]; then
    echo "File ${copy_file} not found!"
    exit 1
fi

error="logs/${model}/${dataset}/${evaluation}/${phase}/error_${flow_embedding}_${modifier}.out"
output="logs/${model}/${dataset}/${evaluation}/${phase}/${phase}_${flow_embedding}_${modifier}.out"

echo "${phase} ${model} ${dataset} ${evaluation} ${flow_embedding} variables:"
echo -e "\tmodel: ${model}"
echo -e "\terror: ${error}"
echo -e "\toutput: ${output}"
echo -e "\ttime: ${time}"
echo -e "\tmem: ${mem}"
echo -e "\ttmp: ${tmp}"
echo -e "\tcopy file: ${copy_file}"

echo -e "\tdataset: ${dataset}" # (ntu, ntu120, ucf101)
echo -e "\tmodel type: ${flow_embedding}" # (base, cnn, abs, avg)
echo -e "\tdilation: ${dilation}"
echo -e "\tevaluation: ${evaluation}" # (CS/CV, CSub/CSet, 1/2/3)
echo -e "\tobservation ratio: ${obs_ratio}"
echo -e "\tdebug: ${debug}"
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



# # TMP TESTING
# data_path="/fred/oz141/ldezoete/MS-G3D/data/ntu/aligned_data/ntu_CS-flowpose_D3_aligned.npz"
#     # -f "${flow_embedding}" \

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
