#!/bin/bash

# Parse command line arguments
dry_run=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset=*)
            dataset="${1#*=}"
            ;;
        --num_experts=*)
            num_experts="${1#*=}"
            ;;
        --train_labels_path=*)
            train_labels_path="${1#*=}"
            ;;
        --start_device=*)
            start_device="${1#*=}"
            ;;
        --end_device=*)
            end_device="${1#*=}"
            ;;
        --num_runs=*)
            num_runs="${1#*=}"
            ;;
        --env_name=*)
            env_name="${1#*=}"
            ;;
        --save_dir=*)
            save_dir="${1#*=}"
            ;;
        --dry_run=*)
            dry_run="${1#*=}"
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
    esac
    shift
done

# Parameters for python script
script_params="--dataset=$dataset --num_experts=$num_experts --train_labels_path $train_labels_path --buffer_path $save_dir"

# Loop through devices and run in tmux sessions
for ((device=$start_device; device<=$end_device; device++)); do
    for ((i=0; i<num_runs; i++)); do
        if [ $dry_run == 0 ]; then
            tmux new-session -d -s "run_${device}_$i"
            tmux send-keys "conda activate $env_name && CUDA_VISIBLE_DEVICES=$device python buffer.py $script_params" C-m
        fi
        echo "tmux new-session -d -s run_${device}_$i"
        echo "tmux send-keys conda activate $env_name && CUDA_VISIBLE_DEVICES=$device python buffer.py $script_params"
    done
done