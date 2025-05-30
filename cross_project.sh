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
SEEDS=(1 2 3 4 5)

for SEED in "${SEEDS[@]}"; do
    echo "Running with data_path: $BASE_DIR and random_seed: $SEED"
    python cross_project.py --data_path "$BASE_DIR" --random_seed "$SEED"
done
