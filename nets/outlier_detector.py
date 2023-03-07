import argparse

import numpy as np
import torch
from sklearn import svm
from torch import nn
from datasets.media_gesture import MediaGestureDataset
from datasets.transformations import BasicTransform, NormalizeMaxSpan, WristAsOrigin


class OutlierDetector:
    def __init__(self, nu: float = 0.05, data_dir: str = "./data", verbose=1):
        # Create a one-class SVM model with the desired kernel
        self.clf = svm.OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
        self.flatten = nn.Flatten()
        dataset = MediaGestureDataset(
            transform=nn.Sequential(
                BasicTransform(),
                NormalizeMaxSpan(1),
                WristAsOrigin(),
            ),
            data_dir=data_dir
        )
        X_train = []
        for feature, _ in dataset:
            X_train.append(feature)
        X_train = torch.stack(X_train, dim=0)
        X_train = self.flatten(X_train)
        if verbose == 1:
            print("fit outlier detector...")
        self.clf.fit(X_train)
        if verbose == 1:
            print("finish fit!")

    def prediction(self, X):
        X = self.flatten(X)
        y_pred = self.clf.predict(X)
        decision_scores = self.clf.decision_function(X)
        probabilities = 100.0 / (1.0 + np.exp(-decision_scores))
        return y_pred, probabilities
