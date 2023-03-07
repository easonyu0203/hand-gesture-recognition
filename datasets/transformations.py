import torch
from torch import nn
import random
import math


class BasicTransform(nn.Module):
    """
    basic transformation perform on feature
    1. don't use z
    2. flip y
    """

    def __init__(self):
        super(BasicTransform, self).__init__()

    def forward(self, feature):
        """
        basic transformation perform on feature
        1. don't use z
        2. flip y
        """
        # dont use z
        feature = feature[:, [0, 1]]
        # flip y (mediapipe top-left origin, hence we flip so we can view as bottom-left)
        feature[:, 1] = 1 - feature[:, 1]
        return feature


class NormalizeMaxSpan(nn.Module):
    """
    normalize (x, y points have span around max_span, the farthest points have a range of max_span)
    normalize x, y span range to max_span
    """

    def __init__(self, max_span=1):
        super(NormalizeMaxSpan, self).__init__()
        self.max_span = max_span

    def forward(self, feature):
        """
        normalize (x, y points have span around 1, the farthest points have a range of 1)
        normalize x, y span range to 1
        :param feature: tensor[21,2]
        :return: tensor[21,2]
        """
        # normalize (x, y points have span around 1, the farthest points have a range of 1)
        x, y = feature[:, 0], feature[:, 1]
        x_span, y_span = torch.max(x) - torch.min(x), torch.max(y) - torch.min(y)
        feature[:, 0], feature[:, 1] = (x * self.max_span) / x_span, (y * self.max_span) / y_span
        return feature


class WristAsOrigin(nn.Module):
    """
    let wrist point to be (0, 0)
    """

    def __init__(self):
        super(WristAsOrigin, self).__init__()

    def forward(self, feature):
        """
        let wrist point to be (0, 0)
        :param feature: tensor[21,2]
        :return: tensor[21,2]
        """
        feature = feature - feature[0]
        return feature


class RandomMirror(nn.Module):
    """
    let wrist point to be (0, 0)
    """

    def __init__(self, probability=0.5):
        self.probability = probability
        super(RandomMirror, self).__init__()

    def forward(self, feature):
        """
        flip sign for x
        :param feature: tensor[21,2]
        :return: tensor[21,2]
        """
        if random.random() < self.probability:
            feature[:, 0] = -feature[:, 0]
        return feature


class RandomRotate(torch.nn.Module):
    def __init__(self, angle_range):
        super(RandomRotate, self).__init__()
        self.angle_range = angle_range

    def forward(self, input_tensor):
        # Get the axis of rotation from the first data point
        axis = input_tensor[0]

        # Calculate a random rotation angle within the specified range
        angle_degrees = torch.empty(1).uniform_(-self.angle_range, self.angle_range)
        # Convert the angle to radians
        angle_radians = torch.tensor(math.radians(angle_degrees.item()))

        # Define the rotation matrix
        cos_theta = torch.cos(angle_radians)
        sin_theta = torch.sin(angle_radians)
        rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
                                        [sin_theta, cos_theta]])

        # Apply the rotation to the input tensor
        rotated_tensor = torch.mm((input_tensor - axis), rotation_matrix) + axis

        return rotated_tensor
