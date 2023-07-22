# some basic imports

import math
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from torchvision import transforms


import dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Block(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # deconvolution
        self.dec_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # decoder blocks
        self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()) # 1x1 convolution for producing the output

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x)  # reduce to 1 channel


def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.stack([np.array(Image.open(f)) for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.


def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)


class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, path, device, use_patches=False, resize_to=(400, 400)):
        self.path = path
        self.device = device
        self.use_patches = use_patches
        self.resize_to=resize_to
        self.x, self.y, self.n_samples = None, None, None
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(os.path.join(self.path, 'images'))[:,:,:,:3]
        self.y = load_all_from_path(os.path.join(self.path, 'groundtruth'))
        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        return x, y

    def __getitem__(self, item):
        return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))
    
    def __len__(self):
        return self.n_samples


if __name__ == '__main__':
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model and move it to the GPU if available
    model = UNet().to(device)

    # Specify a loss function and an optimizer
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # # # Suppose we have a dataloader that gives us batches of images and corresponding labels
    # # # For the sake of this example, we will create a simple random dataloader
    # # def random_dataloader(num_batches, batch_size, image_size=(3, 400, 400)):
    # #     for _ in range(num_batches):
    # #         images = torch.randn(batch_size, *image_size)
    # #         labels = torch.randint(0, 2, (batch_size, 1)).float()
    # #         yield images.to(device), labels.to(device)

    # # dataloader = random_dataloader(num_batches=1000, batch_size=32)
    # original_dataset = dataloader.LazyImageDataset(
    #     'Datasets/ethz-cil-road-segmentation-2023/metadata.csv')
    # loader = DataLoader(original_dataset, 32, shuffle=True)
    train_dataset = ImageDataset('data/training', 'cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset = ImageDataset('data/validation', 'cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

    num_epochs = 35

    # original_dataset = dataloader.LazyImageDataset(
    #     'Datasets/ethz-cil-road-segmentation-2023/metadata.csv')
    # loader = DataLoader(original_dataset, 32, shuffle=True)

    # # Training loop
    # model.train()  # Put the model in training mode
    # for epoch in range(num_epochs):
    #     for i, (image, label) in enumerate(loader):

    #         image = image.to(device)
    #         label = label.to(device)
    #         # Forward pass
    #         outputs = model(image)
    #         loss = loss_function(outputs, label)

    #         # Backward pass and optimization
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         print(loss.item())

    for epoch in range(num_epochs):
        model.train()  # Put the model in training mode
        running_loss = 0.0
        for i, (image, label) in enumerate(train_dataloader):
            # resize = transforms.Resize((224, 224))
            # image = resize(image)
            # label = resize(label)

            image = image.to(device)
            label = label.to(device)
            # Forward pass
            outputs = model(image)
            loss = loss_function(outputs, label)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f'Epoch {epoch+1}, Batch {i+1}, Average Loss: {running_loss / 50}')
            running_loss = 0.0

        # # Evaluate on the validation set
        # model.eval()  # Put the model in evaluation mode
        # with torch.no_grad():
        #     val_labels = []
        #     val_outputs = []
        #     for images, labels in val_dataloader:
        #         outputs = model(images)
        #         val_labels.append(labels.cpu().numpy())
        #         val_outputs.append(outputs.cpu().numpy())

        #     # Concatenate all the outputs and labels
        #     val_labels = np.concatenate(val_labels)
        #     val_outputs = np.concatenate(val_outputs)

        #     # Compute the F1 score
        #     val_f1 = f1_score(val_labels, val_outputs.round())

        print(f'End of Epoch {epoch+1}/{num_epochs}') #, Validation F1: {val_f1}')

    torch.save(model, 'model/almost_my_transformer_with_proper_unet.pt')
