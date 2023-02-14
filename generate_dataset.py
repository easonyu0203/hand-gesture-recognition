import os.path

import cv2
import mediapipe as mp
import numpy as np
import argparse

from utils.CV_Draw import Draw
from nets.my_holistic import MyHolistic


def get_args():
    """get cli args"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0, help="webcam device index")
    parser.add_argument("--gesture", help='gesture to be generate', type=str, default="None")
    parser.add_argument("--min_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Arguments parsing ###############################################################
    args = get_args()
    cap_device = args.device
    gesture_name = args.gesture
    min_confidence = args.min_confidence
    mp_holistic = mp.solutions.holistic
    data_file_path = os.path.join(os.getcwd(), "data", f"{gesture_name}.npy")

    # save data system init #####################################################################
    data = []
    dir_path = os.path.dirname(data_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # append to old data
    old_dataset_size, data_old = 0, None
    if os.path.exists(data_file_path):
        data_old = np.load(data_file_path)
        old_dataset_size = len(data_old)

    # Model Initialize ###############################################################
    model = MyHolistic(min_confidence=0.5)

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
