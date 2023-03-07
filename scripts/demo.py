import torch
import cv2
import argparse
from nets.my_holistic import MyHolistic
from nets.hand_ges_rec_net import HandGesRecNet
from utils.CV_Draw import Draw

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
    # Arguments parsing ###############################################################
    cap_device = args.device
    hand_net_path = args.hand_net_path

    # set up net #####################################################################
    hand_ges_rec_net: HandGesRecNet = torch.load(hand_net_path)
    holistic = MyHolistic()

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
        result, holistic_result = net_pipeline_process(image, holistic, hand_ges_rec_net)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw Result ##############################################################
        draw = Draw()
        image = draw.draw_hand_landmark(holistic_result, image)
        if result["left"]:
            name, conf = result["left"]["name"], result["left"]["confidence"]
            image = draw.draw_box(holistic_result.left_np, image, title=f"{name} {conf:<2f}")
        if result["right"]:
            name, conf = result["right"]["name"], result["right"]["confidence"]
            image = draw.draw_box(holistic_result.right_np, image, title=f"{name} {conf:<2f}")

        # Key press ##############################################################
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # esc
            break

        # show ##############################################################
        cv2.imshow('MediaPipe Holistic', image)


def hand_net_process(hand_ges_rec_net, landmarks):
    # transform landmarks for hand_ges_rec_net
    landmarks = torch.from_numpy(landmarks).type(torch.float)
    features = hand_ges_rec_net.transform(landmarks)
    # net make prediction
    pred_label_name, confidence = hand_ges_rec_net.predict(features.unsqueeze(0))
    # TODO: use one class SVM to check in outlier
    
    return pred_label_name[0], confidence[0]


def net_pipeline_process(image, holistic: MyHolistic, hand_ges_rec_net: HandGesRecNet):
    # holistic process
    holistic_result = holistic.process(image)
    # hand net process
    left_result, right_result = None, None
    if holistic_result.left_hand_landmarks:
        pred_label_name, confidence = hand_net_process(hand_ges_rec_net, holistic_result.left_np)
        left_result = {"name": pred_label_name, "confidence": confidence.item()}
    if holistic_result.right_hand_landmarks:
        pred_label_name, confidence = hand_net_process(hand_ges_rec_net, holistic_result.right_np)
        right_result = {"name": pred_label_name, "confidence": confidence.item()}

    return {"left": left_result, "right": right_result}, holistic_result


if __name__ == '__main__':
    main()
