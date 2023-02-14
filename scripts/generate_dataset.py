import os.path
import cv2
import numpy as np
import argparse
from utils.CV_Draw import Draw
from nets.my_holistic import MyHolistic

"""This script captures live video input from a camera device, processes the video frames using a machine learning 
model, and saves the processed data to a NumPy array. The script uses the MediaPipe Holistic model to perform hand 
tracking and gesture recognition on the video frames. The user can specify the camera device to use and the name of 
the gesture to recognize by setting the cap_device and gesture_name variables at the beginning of the script.

As the script runs, it displays the video frames on screen, with the recognized hand landmarks and bounding boxes 
overlaid on the image. The user can press the spacer to save the current hand landmark data to a NumPy array, 
which is stored in a file specified by the data_file_path variable. The script also displays the number of recorded 
samples on the video frame in real time.

If a previous dataset already exists in the specified data file, the script loads it into memory and appends the new 
recorded data to it. Otherwise, it creates a new dataset and saves it to the file. The final dataset is stored as a 
NumPy array of shape (num_samples, num_landmarks, 3), where num_samples is the total number of recorded samples and 
num_landmarks is the number of hand landmarks detected by the MediaPipe Holistic model. After the script finishes 
running, the user can use this dataset to train a machine learning model for gesture recognition."""

parser = argparse.ArgumentParser(description="Hand Gesture Generation")

parser.add_argument("--cap_device", type=int, default=0, help="Index of the webcam device to use (default: 0)")
parser.add_argument("--gesture", type=str, default=None, help="Gesture to generate (default: None)")
parser.add_argument("--data-folder", type=str, default="data", help="Root directory for storing data (default: data)")

args = parser.parse_args()


def main():
    # Arguments parsing ###############################################################
    cap_device = args.cap_device
    gesture_name = args.gesture
    data_file_path = os.path.join(os.getcwd(), args.data_folder, f"{gesture_name}.npy")

    # save data system init #####################################################################
    data = []
    old_dataset_size, data_old = 0, None
    if os.path.exists(data_file_path):
        data_old = np.load(data_file_path)
        old_dataset_size = len(data_old)

    # Model Initialize ###############################################################
    model = MyHolistic()

    # Start video input ###############################################################
    cap = cv2.VideoCapture(cap_device)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Flip the image horizontally for a selfie-view display.
        image = cv2.flip(image, 1)
        # process image ##############################################################
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        holistic_result = model.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # visualize result ##############################################################
        draw = Draw()
        image = draw.draw_hand_landmark(holistic_result, image)
        image = draw.draw_hands_box(holistic_result.left_np, holistic_result.right_np, image)
        image = draw.draw_log(f"gesture: {gesture_name}", image)
        image = draw.draw_log(f"record cnt: {old_dataset_size + len(data)}", image)

        # save record when press space ###################################################
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # esc
            break
        elif key == 32:  # space
            image = draw.draw_log(f"SAVE!!", image)
            if holistic_result.left_np is not None:
                data.append(holistic_result.left_np)
            elif holistic_result.right_np is not None:
                data.append(holistic_result.right_np)

        cv2.imshow('MediaPipe Holistic', image)

    # save data #####################################################################
    # append to old data
    if data_old is not None and data:
        data_old = np.load(data_file_path)
        data = np.concatenate((data_old, data))
    # save
    data = np.array(data)
    if data.any():
        np.save(data_file_path, data)
        print(f"save data successfully: {data_file_path}")

    # dispose objects ##############################################################
    cap.release()
    exit(0)


if __name__ == '__main__':
    main()
