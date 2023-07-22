# some basic imports
import argparse

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
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms

from timm.models.vision_transformer import VisionTransformer
import timm

import dataloader

from transformer_with_unet import ViTUNet, UnetDecoder
import torch
from torch import nn
import timm


import torch
from torch import nn
import timm
from torchvision.transforms import Resize

class PixelViT(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch8_224', patch_size=8):
        super().__init__()

        self.patch_size = patch_size

        # Load the ViT model, but remove the classification head
        vit_model = timm.create_model(vit_model_name, pretrained=True, num_classes=0)
        vit_model.head = nn.Identity()

        self.resize = Resize((224, 224))
        self.vit = vit_model
        
        self.upsample = nn.Upsample(size=(400, 400), mode='nearest')
        self.classifier = nn.Sequential(
            nn.Conv2d(vit_model.embed_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.resize(x)
        
        # Make sure to extract all hidden states
        features = self.vit.forward_features(x)
        x = features[:, 1:]  # Exclude the CLS token

        x = x.permute(0, 2, 1)  # Shape: (batch_size, embed_dim, seq_len)
        seq_len = x.size(2)
        x = x.view(x.size(0), x.size(1), int(np.sqrt(seq_len)), int(np.sqrt(seq_len)))  # Reshape to: (batch_size, embed_dim, height, width)
        x = self.upsample(x)  # Upsample to the original image size
        x = self.classifier(x)  # Classify each pixel
        return x


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


def main(args):
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args['valid']:
        # Create the model and move it to the GPU if available
        model = PixelViT().to(device)

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
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

        num_epochs = 200

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

        torch.save(model, 'model/just_a_tranformer.pt')
    elif args['valid']:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load your saved model
        model = torch.load('model/just_a_tranformer.pt', map_location=torch.device('cpu'))

        train_dataset = ImageDataset('data/training', 'cuda' if torch.cuda.is_available() else 'cpu')
        val_dataset = ImageDataset('data/validation', 'cuda' if torch.cuda.is_available() else 'cpu')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

        y_true = []
        y_pred = []

        with torch.no_grad():
            for i, (image, label) in enumerate(val_dataloader):
                # resize = transforms.Resize((224, 224))
                # image = resize(image)
                # label = resize(label)

                image = image.to(device)
                label = label.to(device)

                # image = image.to(device)
                # label = label.view(-1).to(device)

                outputs = model(image)

                # Apply a threshold of 0.5: above -> 1, below -> 0
                preds = outputs # (outputs > 0.5).float()
                # print(torch.nonzero(preds))
                np_preds = np.squeeze(preds.numpy())
                np_label = np.squeeze(label.numpy())
                np_image = np.transpose(np.squeeze(image.numpy()), (1, 2, 0))
                print(np_image.shape)

                # preds = preds.view(-1)

                y_true.extend(label.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                print(label.shape, preds.shape)
                
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                # Display np_image1
                axs[0].imshow(np_image, cmap='gray')
                axs[0].axis('off')

                # Display np_image2
                axs[1].imshow(np_label, cmap='gray')
                axs[1].axis('off')

                # Display np_image3
                axs[2].imshow(np_preds, cmap='gray')
                axs[2].axis('off')
                # Save the figure with both subplots
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.tight_layout()
                plt.savefig(f'preds/combined_{i}.png', bbox_inches='tight', pad_inches=0)
                plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid')
    args = parser.parse_args()
    
    main(vars(args))