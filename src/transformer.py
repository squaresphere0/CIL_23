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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader

import dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, E, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x

class ViT(nn.Module):
    def __init__(self, in_chans=3, patch_size=16, embed_dim=512, num_heads=8, num_layers=12, num_outputs=1):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_chans, embed_dim)

        encoder_layers = TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        self.classifier = nn.Conv2d(embed_dim, num_outputs, kernel_size=1)
        self.upsample = nn.Upsample((400, 400), mode='bilinear')  # Add an upsampling layer

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.classifier(x)
        x = self.upsample(x)  # Upsample to match the size of your labels
        return torch.sigmoid(x)  # Sigmoid activation for binary clas

if __name__ == '__main__':
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model and move it to the GPU if available
    model = ViT().to(device)

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

    for i, (image, label) in enumerate(train_dataloader):
        image = image.to(device)
        label = label.to(device)
        print(image.shape)
        raise SystemExit

    num_epochs = 30

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

    torch.save(model, 'model/almost_my_transformer.pt')
