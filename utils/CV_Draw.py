from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np

# Constant
box_padding = (3, 3)  # (x %, y %) relative to the box width, height
box_color = (0, 0, 0)  # BGR
box_thickness = 3  # px

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class Draw:
    log_line_cnt: int

    def __init__(self):
        self.log_line_cnt = 0
        pass

    @staticmethod
    def draw_hand_landmark(result: NamedTuple, image):
        for hand_landmarks in [result.left_hand_landmarks, result.right_hand_landmarks]:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
        return image

    @staticmethod
    def draw_hands_box(left, right, image):
        if left is not None:
            image = Draw.draw_box(left, image, f"left")
        if right is not None:
            image = Draw.draw_box(right, image, f"right")
        return image

    @staticmethod
    def draw_box(landmarks, image, title="hello"):
        x_max, y_max, _ = np.max(landmarks, axis=0)
        x_min, y_min, _ = np.min(landmarks, axis=0)

        image_height, image_width, _ = image.shape
        x_max, x_min = int(x_max * image_width), int(x_min * image_width)
        y_max, y_min = int(y_max * image_height), int(y_min * image_height)
        # considerate padding
        x_pad, y_pad = ((x_max - x_min) * box_padding[0]) // 100, ((y_max - y_min) * box_padding[1]) // 100
        x_max, x_min = x_max + x_pad, x_min - x_pad
        y_max, y_min = y_max + y_pad, y_min - y_pad

        # draw bounding box
        image = cv2.rectangle(image,
                              (x_min, y_min),
                              (x_max, y_max),
                              box_color,
                              box_thickness)
        # draw title
        if title is not None:
            image = cv2.putText(image, title, (x_min, y_min - y_pad),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (1, 1, 1), 3, bottomLeftOrigin=False
                                )

        return image

    def draw_log(self, text, image):
        height = image.shape[0]
        gap = height * 0.04
        cv2.putText(image, text, (int(height * 0.01), int(height * 0.04 + gap * self.log_line_cnt)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (1, 1, 1), 3, bottomLeftOrigin=False
                    )
        self.log_line_cnt += 1
        return image
