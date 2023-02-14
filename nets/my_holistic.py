from typing import NamedTuple

import mediapipe as mp
from mediapipe.python.solutions.holistic import Holistic
import numpy as np

mp_holistic = mp.solutions.holistic


class MyHolistic:
    """
    A wrapper class for dealing with mediapipe holistic
    """
    holistic: Holistic

    def __init__(self, min_confidence=0.5):
        """
        A wrapper class for dealing with mediapipe holistic
        :param min_confidence:
        """
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=min_confidence)

    def process(self, image):
        """
        process image to get landmark and recognize gesture
        :param image: image
        :return:
        """
        holistic_result: NamedTuple = self.holistic.process(image)
        # flip hand (mirror image input)
        holistic_result.right_hand_landmarks, holistic_result.left_hand_landmarks = \
            holistic_result.left_hand_landmarks, holistic_result.right_hand_landmarks
        (holistic_result.left_np, holistic_result.right_np) = MyHolistic._get_hand_landmarks_np(holistic_result)

        return holistic_result

    def __del__(self):
        self.holistic.close()

    @staticmethod
    def _get_hand_landmarks_np(result):
        """
        get numpy array from holistic result
        :param result: the result from holistic models
        :return: (left, right) numpy array
        """
        left, right = None, None
        if result.left_hand_landmarks is not None:
            left = result.left_hand_landmarks.landmark
            left = np.array([[land.x, land.y, land.z] for land in left])
        if result.right_hand_landmarks is not None:
            right = result.right_hand_landmarks.landmark
            right = np.array([[land.x, land.y, land.z] for land in right])
        return left, right
