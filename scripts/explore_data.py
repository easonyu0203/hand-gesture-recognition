import argparse
import numpy as np
import os

"""
This is a Python script that can be used to explore a collection of numpy arrays stored in separate .npy files, 
with each numpy array representing a different class for a classification task. The script takes a single command 
line argument, which is the path to the folder containing the numpy arrays. When the script is run, it loads each of 
the numpy arrays in the folder and prints out the class name, the number of records in that class, and the mean and 
standard deviation of the data for that class.

To use the script, you should first make sure that you have numpy installed, as the script depends on the numpy 
library. You can then run the script from the command line by typing:
python explore_data.py /path/to/data/folder

where /path/to/data/folder is the path to the folder containing the numpy arrays that you want to explore. When you 
run the script, it will print out information about the structure of the data, which can be useful for exploratory 
data analysis in a machine learning project."""


def main(data_folder):
    classes = []

    # Get a list of all the .npy files in the data folder
    npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]

    # Loop through each .npy file and load the data
    for npy_file in npy_files:
        class_name = os.path.splitext(npy_file)[0]
        classes.append(class_name)
        npy_data = np.load(os.path.join(data_folder, npy_file))
        num_records = npy_data.shape[0]
        print(f"Class: {class_name}, Num Records: {num_records}")

    print(f"Classes: {classes}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print info about numpy arrays in a folder.')
    parser.add_argument('--data-folder', type=str, default='data',
                        help='Path to the folder containing numpy arrays (default: data)')
    args = parser.parse_args()
    main(args.data_folder)
