import argparse
import os.path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.tensorboard_helper import write_loss, write_model_pred_figures
from datasets.media_gesture import MediaGestureDataset
from datasets.transformations import BasicTransform, NormalizeMaxSpan, WristAsOrigin
from nets.hand_ges_rec_net import HandGesRecNet
from utils.train.early_stopper import EarlyStopper
from tqdm import tqdm
import copy

"""This script trains a PyTorch neural network for hand gesture recognition using the MediaPipe dataset. The script 
takes several command line arguments, including the batch size, learning rate, number of epochs, and early stopping 
parameters. The script also specifies the path where the trained model will be saved.

The main function sets up the data pipeline by creating training and validation datasets from the 
MediaGestureDataset, which is processed with various transformations to standardize the data. A HandGesRecNet model 
is defined with a specified number of features and classes. The loss function is defined as a cross-entropy loss, 
and the optimizer is set as stochastic gradient descent. The EarlyStopper function is used to monitor the validation 
loss and stop training when the validation loss stops improving.

The train_loop function defines the forward pass and backpropagation for training the model, while val_loop function 
calculates the test loss and accuracy for the validation dataset. The main function then trains the model and saves 
it at the specified path when training is complete.

The script also includes utility functions for writing training loss and accuracy to a TensorBoard summary, 
and for writing predicted output figures to the summary for every 500 epochs.

To use the script, you can run it from the command line using the following command:

`python train.py --batch-size 32 --lr 0.001 --epochs 20000 --patience 100 --patience-delta 0.00001 --data-dir ./data
--save-path ./trained_nets/hand_ges_rec_net`

You can adjust the values of the command line arguments to suit your needs.
You can also monitor training process using 
`tensorboard --logdir=runs`
"""


parser = argparse.ArgumentParser(description="Hand Gesture Recognition Training")
parser.add_argument("--batch-size", type=int, default=32, help="batch size for training (default: 32)")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for training (default: 0.001)")
parser.add_argument("--epochs", type=int, default=20000, help="number of epochs to train (default: 20000)")
parser.add_argument("--patience", type=int, default=100,
                    help="number of epochs to wait before early stopping (default: 100)")
parser.add_argument("--patience-delta", type=float, default=0.00001,
                    help="minimum change in validation loss to qualify as improvement for early stopping ("
                         "default: 0.00001)")
parser.add_argument("--data-dir", type=str, default="./data",
                    help="dir for dataset (default: ./data)")
parser.add_argument("--save-path", type=str, default="./trained_nets/hand_ges_rec_net",
                    help="path to save trained model (default: ./trained_nets/hand_ges_rec_net)")
args = parser.parse_args()


def main():
    # Assign parsed arguments to variables
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    patience = args.patience
    patience_delta = args.patience_delta
    save_path = args.save_path
    data_dir = args.data_dir

    # set up
    dataset = MediaGestureDataset(
        transform=nn.Sequential(
            BasicTransform(),
            NormalizeMaxSpan(1),
            WristAsOrigin()
        ),
        data_dir=data_dir
    )
    train_dataset, valid_dataset = random_split(dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    writer = SummaryWriter()
    model = HandGesRecNet(
        feature_cnt=dataset.feature_cnt, class_cnt=dataset.class_cnt,
        transform=dataset.transform, label_idx_to_name=dataset.label_idx_2_name
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    early_stopper = EarlyStopper(patience=patience, delta=patience_delta)

    # training
    best_model = copy.deepcopy(model)
    best_test_loss = float("inf")
    writer.add_graph(model, input_to_model=next(iter(train_dataloader))[0])
    with tqdm(total=epochs, desc="Training") as pbar:
        for t in range(epochs):
            train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
            val_loss, accuracy = val_loop(valid_dataloader, model, loss_fn)
            # save best model
            if val_loss <= best_test_loss:
                best_model, best_test_loss = copy.deepcopy(model), val_loss
            # write to board
            write_loss(writer, train_loss, val_loss, t + 1)
            if t % 500 == 0:
                write_model_pred_figures(writer, model, dataset, 4, t + 1, figsize=6)
            # check early stop
            if early_stopper(val_loss):
                print("Early stopping triggered.")
                break
            # Print train and test metrics
            pbar.update(1)
            pbar.set_postfix(train_loss=f"{train_loss:.5f}", val_loss=f"{val_loss:.5f}",
                             accuracy=f"{accuracy * 100:.2f}%")

    print("Done!")

    # save model
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    torch.save(best_model, save_path)


def train_loop(dataloader, model, loss_fn, optimizer):
    size, epoch_loss = len(dataloader.dataset), 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(X)
    epoch_loss /= size
    return epoch_loss


def val_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item() * len(X)
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= size
    accuracy = correct / size
    return test_loss, accuracy


if __name__ == '__main__':
    main()
