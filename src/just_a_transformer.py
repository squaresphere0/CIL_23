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
from efficientnet_pytorch import EfficientNet


class PixelSwinT(nn.Module):
    def __init__(self, swin_model_name='swin_large_patch4_window12_384'):
        super().__init__()

        # Load the SWIN Transformer model, but remove the classification head
        self.swin = timm.create_model(swin_model_name, pretrained=True, num_classes=0)
        self.swin.head = nn.Identity()

        self.resize = Resize((384, 384))
        
        self.upsample = nn.Upsample(size=(400, 400), mode='bilinear', align_corners=False)
        self.classifier = nn.Sequential(
            nn.Conv2d(1536, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.resize(x)
        
        x = self.swin(x)

        x = x.permute(0, 3, 1, 2)  # permute the dimensions to bring it to (B, Channels, H, W) format
        # x = self.reduce_dim(x)  # reduce dimensionality to 1
        # print(x.shape)
        # x = F.interpolate(x, size=(224, 224))

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


def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    patches_hat = y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    patches = y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    return (patches == patches_hat).float().mean()


def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


def iou_loss_f(pred, target, smooth=1e-6, classes='binary'):
    """
    Compute the Intersection over Union (IoU) loss.

    Parameters:
    pred (Tensor): the model's predictions
    target (Tensor): the ground truth
    smooth (float, optional): a smoothing factor to prevent division by zero
    classes (str, optional): 'binary' for binary segmentation tasks, 'multi' for multi-class tasks

    Returns:
    IoU loss
    """

    # Reshape to ensure the prediction and target tensors are the same shape
    pred = pred.view(-1)
    target = target.view(-1)

    # Compute the intersection
    intersection = (pred * target).sum()

    # Compute the union
    if classes.lower() == 'binary':
        # Binary segmentation -> union = pred + target - intersection
        union = pred.sum() + target.sum() - intersection
    elif classes.lower() == 'multi':
        # Multi-class segmentation -> union = pred + target
        union = pred.sum() + target.sum()
    else:
        raise ValueError(f"'classes' should be 'binary' or 'multi', got {classes}")

    # Compute the IoU and the IoU loss
    iou = (intersection + smooth) / (union + smooth)
    iou_loss = 1 - iou

    return iou_loss


def main(args):
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args['valid']:
        # Create the model and move it to the GPU if available
        model = PixelSwinT().to(device)

        # Specify a loss function and an optimizer
        metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}

        # pos_weight = torch.ones([1, 1, 400, 400])*2.0
        # pos_weight = pos_weight.to(device)
        # bce_loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        bce_loss_function = nn.BCELoss()
        iou_loss_function = accuracy_fn  # This is the function I provided earlier

        bce_weight = 0.8  # This determines how much the BCE loss contributes to the total loss
        iou_weight = 1 - bce_weight  # This determines how much the IoU loss contributes to the total loss        optimizer = torch.optim.Adam(model.parameters())

        optimizer = torch.optim.Adam(model.parameters())
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


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
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

        num_epochs = 200

        for epoch in range(num_epochs):
            model.train()  # Put the model in training mode
            running_loss = 0.0
            # scheduler.step()
            for i, (image, label) in enumerate(train_dataloader):
                # resize = transforms.Resize((224, 224))
                # image = resize(image)
                # label = resize(label)

                image = image.to(device)
                label = label.to(device)
                # Forward pass
                outputs = model(image)
                bce_loss = bce_loss_function(outputs, label)
                iou_loss = iou_loss_function(outputs, label)
                loss = bce_weight * bce_loss + iou_weight * iou_loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            if epoch % 20 == 0:
                torch.save(model, 'model/just_a_tranformer.pt')
            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}, Average Loss: {running_loss / 50}')
            running_loss = 0.0

            if epoch % 20 == 0 and epoch != 0:
                # Evaluate on the validation set
                print("Evaluating, plotting images.")
                model.eval()  # Put the model in evaluation mode
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
                        preds = outputs # (outputs > 0.15).float()
                        # print(torch.nonzero(preds))
                        np_preds = np.squeeze(preds.cpu().numpy())
                        np_label = np.squeeze(label.cpu().numpy())
                        np_image = np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0))
                        # print(np_image.shape)

                        # preds = preds.view(-1)

                        # y_true.extend(label.cpu().numpy())
                        # y_pred.extend(preds.cpu().numpy())
                        # print(label.shape, preds.shape)

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
                        plt.savefig(f'preds/combined_{i}_epoch_{epoch}.png', bbox_inches='tight', pad_inches=0)
                        plt.close()

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
                resize = transforms.Resize((224, 224))
                image = resize(image)
                label = resize(label)

                image = image.to(device)
                label = label.to(device)

                # image = image.to(device)
                # label = label.view(-1).to(device)

                outputs = model(image)

                # Apply a threshold of 0.5: above -> 1, below -> 0
                preds = outputs # (outputs > 0.15).float()
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
