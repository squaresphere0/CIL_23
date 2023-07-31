### THIS CODE IS TAKEN FROM THE PROJECT_3_FINAL.IPYNB NOTEBOOK PROVIDED BY THE COURSE
import torch
from torch import nn

class PatchCNN(nn.Module):
    # simple CNN for classification of patches
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.BatchNorm2d(16),
                                 nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.BatchNorm2d(32),
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.BatchNorm2d(64),
                                 nn.Dropout(0.5),
                                 nn.Flatten(),
                                 nn.Linear(256, 10),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(10, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()
