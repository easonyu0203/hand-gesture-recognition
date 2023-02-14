import torch
import cv2
import argparse
from torch import nn
from nets.my_holistic import MyHolistic
from nets.hand_ges_rec_net import HandGesRecNet
from utils.CV_Draw import Draw


def get_args():
    """get cli args"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0, help="webcam device index")
    parser.add_argument("--hand_net_path", type=str, default="./trained_nets/hand_ges_rec_net", help="path to trained "
                                                                                                     "hand gesture "
                                                                                                     "recognition net")

    args = parser.parse_args()

    return args


def hand_net_process(hand_ges_rec_net, landmarks):
    landmarks = torch.from_numpy(landmarks).type(torch.float)
    features = hand_ges_rec_net.transform(landmarks)
    pred_label_name, confidence = hand_ges_rec_net.predict(features.unsqueeze(0))
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
    # Arguments parsing ###############################################################
    args = get_args()
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
