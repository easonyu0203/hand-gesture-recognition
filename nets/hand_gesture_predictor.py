import cv2
import torch

from nets.my_holistic import MyHolistic
from nets.outlier_detector import OutlierDetector


class HandGesturePredictor:
    def __init__(self, hand_net_path, outlier_nu=0.5):
        """

        :param hand_net_path: path for hand net
        :param outlier_nu: 0~1, nu mean the upper bound of train error for outlier detection
        """
        self.hand_ges_rec_net = torch.load(hand_net_path)
        self.outlier_detector = OutlierDetector(nu=outlier_nu, verbose=1)
        self.holistic = MyHolistic()

    def process(self, image):
        # convert image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # process image ##############################################################
        holistic_result = self.holistic.process(image)
        # hand net process
        left_result, right_result = None, None
        if holistic_result.left_hand_landmarks:
            pred_label_name, confidence, is_outlier, outlier_confidence = self.hand_net_process(holistic_result.left_np)
            left_result = {"name": pred_label_name, "confidence": confidence.item(),
                           "is_outlier": is_outlier, "outlier_confidence": outlier_confidence}
        if holistic_result.right_hand_landmarks:
            pred_label_name, confidence, is_outlier, outlier_confidence = self.hand_net_process(
                holistic_result.right_np)
            right_result = {"name": pred_label_name, "confidence": confidence.item(),
                            "is_outlier": is_outlier, "outlier_confidence": outlier_confidence}

        return {"left": left_result, "right": right_result}, holistic_result

    def hand_net_process(self, landmarks):
        # transform landmarks for hand_ges_rec_net
        landmarks = torch.from_numpy(landmarks).type(torch.float)
        features = self.hand_ges_rec_net.transform(landmarks)
        # net make prediction
        pred_label_name, confidence = self.hand_ges_rec_net.predict(features.unsqueeze(0))
        # outlier detection
        is_outliers, outlier_confidence = self.outlier_detector.prediction(features.unsqueeze(0))

        return pred_label_name[0], confidence[0], is_outliers[0], outlier_confidence[0]
