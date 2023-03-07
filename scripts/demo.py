import torch
import cv2
import argparse

from nets.hand_gesture_predictor import HandGesturePredictor
from utils.CV_Draw import Draw
from utils.cv2.video_input import VideoInput

"""This script is a demo for hand gesture recognition using the MediaPipe and PyTorch libraries. The script takes two 
command line arguments, the index of the webcam device to use and the path to the trained hand gesture recognition 
model.

The main function sets up the neural network for hand gesture recognition and the holistic hand landmark detector 
from MediaPipe. The function then starts the video input stream and continuously captures video frames from the 
specified webcam device. For each frame, the script processes the image to detect hand landmarks and predicts the 
hand gesture using the trained neural network. The script then visualizes the hand landmarks and the predicted hand 
gesture on the video frame.

The script uses the Draw class to draw the landmarks and predicted hand gesture labels and confidence scores on the 
video frame, and it uses OpenCV to display the resulting video stream.

To use the script, you can run it from the command line using the following command:
python demo.py --device 0 --hand-net-path ./trained_nets/hand_ges_rec_net
You can adjust the values of the command line arguments to suit your needs."""

parser = argparse.ArgumentParser(description="Hand Gesture Recognition")
parser.add_argument("--device", type=int, default=0, help="Index of the webcam device to use (default: 0)")
parser.add_argument("--hand-net-path", type=str, default="./trained_nets/hand_ges_rec_net",
                    help="Path to the trained hand gesture recognition model (default: ./trained_nets/hand_ges_rec_net)")
args = parser.parse_args()


def main():
    detector = HandGestureDetector(args.device, args.hand_net_path)
    detector.start()


class HandGestureDetector:
    def __init__(self, device, hand_net_path):
        self.video_input = VideoInput(device)
        self.hand_gesture_predictor = HandGesturePredictor(hand_net_path)
        self.video_input.on_image_processed = self.process_image

    def start(self):
        self.video_input.start()

    def stop(self):
        self.video_input.stop()

    def process_image(self, image):
        result, holistic_result = self.hand_gesture_predictor.process(image)

        # Draw Result ##############################################################
        draw = Draw()
        image = draw.draw_hand_landmark(holistic_result, image)
        if result["left"]:
            name, conf = result["left"]["name"], result["left"]["confidence"]
            is_outlier, outlier_conf = result["left"]["is_outlier"], result["left"]["outlier_confidence"]
            out_str = ("outlier" if is_outlier == -1 else "inlier") + f" {conf:2<f}"
            image = draw.draw_box(holistic_result.left_np, image, title=f"{name} {conf:<2f}, {out_str}")
        if result["right"]:
            name, conf = result["right"]["name"], result["right"]["confidence"]
            is_outlier, outlier_conf = result["right"]["is_outlier"], result["right"]["outlier_confidence"]
            out_str = ("outlier" if is_outlier == -1 else "inlier") + f" {conf:2<f}"
            image = draw.draw_box(holistic_result.right_np, image, title=f"{name} {conf:<2f} {conf:<2f}, {out_str}")

        # Key press ##############################################################
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # esc
            self.stop()

        # show ##############################################################
        cv2.imshow('MediaPipe Holistic', image)


if __name__ == '__main__':
    main()
