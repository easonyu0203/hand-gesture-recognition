import argparse
import asyncio
import cv2

from nets.hand_gesture_predictor import HandGesturePredictor
from nets.utils.draw_result import draw_result
from utils.cv2.video_input import VideoInput

"""This script is a demo for hand gesture recognition using the MediaPipe and PyTorch libraries. The script takes two 
command line arguments, the index of the webcam device to use and the path to the trained hand gesture recognition 
model.

To use the script, you can run it from the command line using the following command:
python demo.py --device 0 --hand-net-path ./trained_nets/hand_ges_rec_net
You can adjust the values of the command line arguments to suit your needs."""

parser = argparse.ArgumentParser(description="Hand Gesture Recognition")
parser.add_argument("--device", type=int, default=0, help="Index of the webcam device to use (default: 0)")
parser.add_argument("--hand-net-path", type=str, default="./trained_nets/hand_ges_rec_net",
                    help="Path to the trained hand gesture recognition model (default: ./trained_nets/hand_ges_rec_net)")
parser.add_argument("--data-dir", type=str, default="./data", help="Path to dataset")
args = parser.parse_args()


async def main():
    video_input = VideoInput(args.device)
    hand_gesture_predictor = HandGesturePredictor(args.hand_net_path, args.data_dir, outlier_nu=0.3)

    # start video input
    asyncio.create_task(video_input.start())

    while True:
        # Wait for a new frame to be available
        image = await video_input.get_frame()
        # process image
        (result, holistic_result) = hand_gesture_predictor.process(image)
        # show result
        image = draw_result(result, holistic_result, image)
        cv2.imshow('MediaPipe Holistic', image)

        # get key
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # esc
            cv2.destroyWindow('MediaPipe Holistic')


if __name__ == '__main__':
    asyncio.run(main())
