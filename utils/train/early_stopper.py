import numpy as np
import torch


class EarlyStopper:
    def __init__(self, patience=10, delta=0, mode='min'):
        """
        Initialize early stopper.

        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            delta (float): Minimum change in monitored quantity to qualify as an improvement.
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the monitored quantity stops decreasing;
                        in 'max' mode it will stop when the quantity stops increasing.
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        if mode == 'min':
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, monitored_quantity):
        """
        Check if training should be stopped early based on the monitored quantity.

        Args:
            monitored_quantity (float): Value of monitored quantity on the current epoch.

        Returns:
            bool: True if training should be stopped early, False otherwise.
        """
        if self.mode == 'min':
            score = -monitored_quantity
        else:
            score = monitored_quantity

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

