# Hand Gesture Recognition

Hand Gesture Recognition using PyTorch and Google Media Pipeline Holistic

# Description

The purpose of this project is to recognize hand gestures using PyTorch and the Google Media Pipeline, and to generate data for training a neural network. The goal is to build a neural network that can recognize different hand gestures accurately, and to train this neural network using data generated from the MediaPipe Holistic model. The project aims to provide a complete pipeline for hand gesture recognition, from data generation to model training to real-time demo, using machine learning techniques.

# What have Achieved

- Implemented a hand gesture recognition pipeline using PyTorch and the Google Media Pipeline Holistic model
- Developed a fully connected neural network, `HandGesRecNet`, that can classify different hand gestures with high accuracy
- creation of a `generate_dataset.py` script that captures live video input, performs hand tracking and gesture recognition using the MediaPipe Holistic model, and saves the processed data as NumPy arrays for use in training the `HandGesRecNet` neural network
- demonstrates the hand gesture recognition pipeline in real-time using a trained model

# Quick Start

1. Clone this repository to your local machine.
2. Install the required libraries using the command: `pip install -r requirements.txt`.
3. To generate datasets, like generate for open gesture and close gesture, run the script `python scripts/generate_dataset.py` with `--gesture open` and `--gesture close` respectfully.
    
    > As the script runs, it displays the video frames on screen, with the recognized hand landmarks and bounding boxes overlaid on the image. You can press the spacer to save the current hand landmark data to a NumPy array. The script also displays the number of recorded samples on the video frame in real time.
4. To check the generated dataset, use `python scripts/explore_data.py` to have an overview of dataset, it should output something like this:
    
    ```
    Class: open, Num Records: 75
    Class: close, Num Records: 77
    Classes: ['open', 'close']
    ```
    
5. To train the `HandGesRecNet` model using the generated dataset, run the script `python scripts/train.py` to monitor training process run `tensorboard --logdir=runs`
6. To run a demonstration of the hand gesture recognition system, run the script `python scripts/demo.py`

# Usage

To use this project, follow the steps below:

### **1. Generating Hand Gesture Data**

Use the **`generate_dataset.py`** script to capture live video input from a camera device, process the video frames using the MediaPipe Holistic model, and save the processed data to a NumPy array. The script saves the captured hand landmarks to a dataset file that can be used to train a machine learning model for hand gesture recognition.

To run the script, use the following command:

```
python scripts/generate_dataset.py --cap_device 0 --gesture_name "thumbs_up"
```

- `cap_device`: Index of the camera device to use. Default is 0.
- `gesture_name`: Name of the hand gesture to recognize. Default is "thumbs_up".

You can change the values of these command-line arguments to specify a different camera device or hand gesture.

### **2. Training Hand Gesture Recognition Model**

Use the `train.py` script to train a PyTorch neural network for hand gesture recognition using the generated dataset.

To run the script, use the following command:

```
python scripts/train.py --batch_size 32 --lr 0.001 --epochs 20000 --patience 100 --patience_delta 0.00001 --save_path ./trained_nets/hand_ges_rec_net
```

- `batch_size`: Number of samples in each batch for training. Default is 32.
- `lr`: Learning rate for the optimizer. Default is 0.001.
- `epochs`: Number of epochs for training. Default is 20000.
- `patience`: Early stopping patience parameter. Default is 100.
- `patience_delta`: Minimum change in validation loss to be considered as an improvement. Default is 0.00001.
- `data_dir`: Path to save the trained model. Default is `./data`
- `save_path`: Path to save the trained model. Default is `./trained_nets/hand_ges_rec_net`.

You can adjust these parameters to suit your needs.

### **3. Hand Gesture Recognition Demo**

Use the `demo.py` script to run a demo for hand gesture recognition in real-time using the trained model.

To run the script, use the following command:

```
python scripts/demo.py --device 0 --hand_net_path ./trained_nets/hand_ges_rec_net
```

- `device`: Index of the camera device to use. Default is 0.
- `hand_net_path`: Path to the trained hand gesture recognition model. Default is `./trained_nets/hand_ges_rec_net`.

You can adjust these parameters to suit your needs.

### **4. Explore Project Folders**

The project also contains the following directories that might be useful:

- `Data`: Contains NumPy arrays for each class of hand gesture, where each file represents a specific gesture class and contains a numpy array of [num_record, num_landmark, 3].
- `datasets`: Contains the dataset class for PyTorch model training.
- `nets`: Contains the PyTorch model for hand gesture recognition.
- `Scripts`: Contains scripts for generating gesture data, training the model, and running a demo.
- `Utils`: Contains helper functions used in the project.

Explore these directories to learn more about the project and its inner workings.

Regenerate response

# Folder Structure

```
├── Data: (generated by generate_dataset.py)
│   ├── class1.npy
│   ├── class2.npy
│   ├── ...
│   └── classN.npy
├── datasets
│   ├── MediaGestureDataset.py
│   └── ...
├── nets
│   ├── HandGesRecNet.py
│   └── ...
├── Scripts
│   ├── generate_dataset.py
│   ├── train.py
│   └── demo.py
├── Utils
│   ├── helper_functions.py
│   └── ...
├── README.md
└── .gitignore
```

The `Data` directory contains NumPy arrays for each class of hand gesture, where each file represents a specific gesture class and contains a numpy array of [num_record, num_landmark, 3].

The `datasets` directory contains the dataset class for PyTorch model training.

The `nets` directory contains the PyTorch model for hand gesture recognition `HandGesRecNet`.

The `Scripts` directory contains the following scripts:

- `generate_dataset.py`: script to generate gesture data using the webcam and the MediaPipe Holistic model
- `train.py`: script to train the `HandGesRecNet` model using the generated data
- `demo.py`: script to run the pipeline for hand gesture recognition in real-time using the trained model.

The `Utils` directory contains helper functions used in the project.

The `README.md` file provides an overview of the project and its directory structure, and the `.gitignore` file contains the files and directories that will not be tracked by Git.

# Project Asset

- Github
    
    [https://github.com/easonyu0203/hand-gesture-recognition](https://github.com/easonyu0203/hand-gesture-recognition)