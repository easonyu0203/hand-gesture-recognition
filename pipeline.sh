#!/bin/sh
export PYTHONPATH=$(pwd)

# Check if the list of gesture names is provided as an argument
if [ $# -eq 0 ]
then
  echo "Error: List of gesture names not provided."
  echo "Usage: ./run_script.sh <gesture_name_1> <gesture_name_2> ... <gesture_name_n>"
  exit 1
fi

# Set the list of gesture names from the command-line arguments
name_list=("$@")

# Remove all data, run, and trained_net
./rm_all_data_run_model.sh

# Loop through the list and run the generate_dataset.py script with each gesture name
for name in "${name_list[@]}"
do
    python scripts/generate_dataset.py --gesture "$name"
done

# Check the dataset
python scripts/explore_data.py

# Train the model
python scripts/train.py

# Demo the model
python scripts/demo.py
