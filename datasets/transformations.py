import torch
from torch import nn


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
