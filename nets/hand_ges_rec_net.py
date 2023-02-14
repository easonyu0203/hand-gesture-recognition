import torch
from torch import nn


class HandGesRecNet(nn.Module):
    """
    net for hand gesture recognition using mediapipe landmarks as input
    """

    def __init__(self, feature_cnt, class_cnt, transform, label_idx_to_name=None):
        """
        net for hand gesture recognition using mediapipe landmarks as input
        :param feature_cnt: feature_cnt
        :param class_cnt: class_cnt
        :param transform: transform
        :param label_idx_to_name:
        """
        super(HandGesRecNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(feature_cnt, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, class_cnt)
        )
        self.label_idx_to_name = label_idx_to_name
        self.transform = transform
        self.feature_cnt = feature_cnt
        self.class_cnt = class_cnt

    def forward(self, x):
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

    def predict(self, features):
        with torch.no_grad():
            preds = self(features)
            pred_idxs = preds.argmax(1)
            confidence = torch.gather(nn.Softmax(dim=1)(preds), 1, pred_idxs.unsqueeze(1)) * 100
            pred_label_name = [self.label_idx_to_name[i.item()] for i in pred_idxs]
        return pred_label_name, confidence
