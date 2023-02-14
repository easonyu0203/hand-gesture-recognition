import matplotlib.pyplot as plt
import torch

from utils.mat_draw import draw_hand
from torch import nn
from datasets.media_gesture import MediaGestureDataset
from random import randint


def write_loss(writer, train_loss, test_loss, epoch):
    writer.add_scalar("Train loss", train_loss, epoch)
    writer.add_scalar("Test loss", test_loss, epoch)


def write_model_pred_figures(writer, model, dataset: MediaGestureDataset, cnt, epoch, figsize=6):
    fig = plt.figure(figsize=(figsize * cnt, figsize))

    # choose samples
    features, labels = [], []
    for _ in range(cnt):
        idx = randint(0, len(dataset) - 1)
        f, l = dataset[idx]
        features.append(f)
        labels.append(l)
    features = torch.stack(features)
    labels = torch.stack(labels)

    # make prediction
    preds = model(features)
    pred_idxs = preds.argmax(1)
    labels_idxs = labels.argmax(1)
    for i, (feature, pred, pred_idx, label_idx) in enumerate(zip(features, preds, pred_idxs, labels_idxs)):
        ax = fig.add_subplot(1, cnt, i + 1)
        draw_hand(feature)
        correct_label = dataset.label_idx_2_name[label_idx.item()]
        pred_label = dataset.label_idx_2_name[pred_idx.item()]
        confidence = nn.Softmax(dim=0)(pred)[pred_idx] * 100
        ax.set_title(f"predict: {pred_label}, correct: {correct_label} confidence: {confidence:>0.1f}%")

    writer.add_figure("test output", fig, global_step=epoch)
