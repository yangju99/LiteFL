#!/bin/bash

# Base directory for processed data (modify this path as needed)
BASE_DIR="./chunks"

# List of dataset filenames
DATASETS=(
    Chart.pkl
    Lang.pkl
    Math.pkl
    Time.pkl
    Closure.pkl
)



# List of random seeds to use
SEEDS=(1 2 3 4)

# Iterate through the dataset filenames and random seeds
for FILE in "${DATASETS[@]}"; do
    DATA_PATH="$BASE_DIR/$FILE"
    for SEED in "${SEEDS[@]}"; do
        echo "Running with data_path: $DATA_PATH and random_seed: $SEED"
        python leave_one_out_project.py --data_path "$DATA_PATH" --random_seed "$SEED"
    done
done