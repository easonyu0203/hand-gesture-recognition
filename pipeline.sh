#!/bin/sh
export PYTHONPATH=$(pwd)

# this script will generate dataset => train model => demo it

# Define the list of gesture names
# shellcheck disable=SC2039
name_list=("shoot" "grab" "hold")

# remove all data, run, trained_net
./rm_all_data_run_model.sh

# Loop through the list and run the generate_dataset.py script with each gesture name
# shellcheck disable=SC2039
for name in "${name_list[@]}"
do
    python scripts/generate_dataset.py --gesture "$name"
done

# check dataset
python scripts/explore_data.py

# train
python scripts/train.py

# demo
python scripts/demo.py