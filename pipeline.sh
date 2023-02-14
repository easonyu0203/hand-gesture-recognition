#!/bin/sh
export PYTHONPATH=$(pwd)

# this script will generate dataset => train model => demo it

# Define the list of gesture names
name_list=("one" "two" "three")

# remove all data, run, trained_net
rm -rf data
rm -rf trained_nets
rm -rf runs

# Loop through the list and run the generate_dataset.py script with each gesture name
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