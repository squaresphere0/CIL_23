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
from io import BytesIO
import cairosvg
import random

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import torchvision
from torchview import draw_graph

import tempfile
import os

import requests
import json

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

        # self.dropout = nn.Dropout(p=0.5)

        self.reduce_channels = nn.Conv2d(1536, 1, kernel_size=1)
        
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1536, out_channels=768, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(768),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=768, out_channels=384, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.Conv2d(in_channels=48, out_channels=1, kernel_size=1),  # Output layer, now with 1 channel
        )
        self.upsample = nn.Upsample(size=(400, 400), mode='bicubic') #, align_corners=True)
        self.classifier = nn.Sequential(
            # nn.Conv2d(1536, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.resize(x)
        
        x = self.swin(x)
        # x = self.dropout(x)

        x = x.permute(0, 3, 1, 2)  # permute the dimensions to bring it to (B, Channels, H, W) format
        intermediate = self.reduce_channels(x)
        intermediate = self.upsample(intermediate)
        intermediate = self.classifier(intermediate)
        # x = self.reduce_dim(x)  # reduce dimensionality to 1
        # print(x.shape)
        # x = F.interpolate(x, size=(224, 224))
        # print("SHape after swin:", x.shape)

        x = self.upscale(x)
        x = self.upsample(x)  # Upsample to the original image size
        x = self.classifier(x)  # Classify each pixel
        return x, intermediate


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


def send_message(text):
    url = "https://api.telegram.org/bot6519873169:AAGxxszlbXMh9CQg9L4gK4EIOGVfcOZE2RI/sendMessage"
    headers = {'Content-Type': 'application/json'}
    data = {
        'chat_id': '502129529',
        'text': text,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    # print(response.status_code, response.json())


def send_photo(photo):
    url = "https://api.telegram.org/bot6519873169:AAGxxszlbXMh9CQg9L4gK4EIOGVfcOZE2RI/sendPhoto"
    files = {'photo': photo}
    data = {
        'chat_id': "502129529",
        'disable_notification': True,
    }
    response = requests.post(url, files=files, data=data)


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
    send_message("Loaded to the execution environment.")

    # Fix randomness
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    log_custom_info_at_each_nth_epoch = 20

    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiment = Experiment(
        api_key = "x6UJjWwiy9x4Z3RaBjZ4hEHGk",
        project_name = "cil-23",
        workspace="mrpetrkol"
    )

    # Create the model and move it to the GPU if available
    model = PixelSwinT().to(device)

    initial_weights_name = 'model/initial_swin_weights.pth'
    # if os.path.isfile(initial_weights_name):
    #     model.swin.load_state_dict(torch.load(initial_weights_name))

    # Specify a loss function and an optimizer
    metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}

    # pos_weight = torch.ones([1, 1, 400, 400])*2.0
    # pos_weight = pos_weight.to(device)
    # bce_loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_loss_function = nn.BCELoss()
    iou_loss_function = accuracy_fn  # This is the function I provided earlier

    bce_weight = 1  # This determines how much the BCE loss contributes to the total loss
    iou_weight = 1 - bce_weight  # This determines how much the IoU loss contributes to the total loss        optimizer = torch.optim.Adam(model.parameters())

    optimizer_upscale = torch.optim.Adam(model.upscale.parameters(), lr=0.01)
    # Gather the rest of the model's parameters
    rest_of_model_params = [p for n, p in model.named_parameters() if 'upscale' not in n]
    optimizer_rest = torch.optim.Adam(rest_of_model_params, lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    my_batch_size = 4
    train_dataset = ImageDataset('data/training', 'cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset = ImageDataset('data/validation', 'cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=my_batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    num_epochs = 100

    hyper_params = {
        # "learning_rate": optimizer.param_groups[0]['lr'],
        # "weight_decay": optimizer.param_groups[0]['weight_decay'],
        "num_epochs": num_epochs,
        "batch_size": my_batch_size,
        'bce_weight': bce_weight,
        'iou_weight': iou_weight,
    }
    experiment.log_parameters(hyper_params)
    
    # Visualize the model
    model_graph = draw_graph(model, input_size=(my_batch_size, 3, 400, 400), expand_nested=True)
    # experiment.log_asset('model_graph.png')
    # Create a temporary file
    model_graph_json = model_graph.visual_graph.render(filename='temp_graph', format='svg', cleanup=True)
    cairosvg.svg2png(url='temp_graph.svg', write_to='temp_graph.png')
    with open('temp_graph.png', 'rb') as f:
        image_bytes = f.read()
        experiment.log_asset_data(image_bytes, name='graph.png', overwrite=True)
        # experiment.set_model_graph(model_graph_json)

    send_message("Starting new computation.")
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
            outputs, intermediate = model(image)
            bce_loss = bce_loss_function(outputs, label)
            iou_loss = iou_loss_function(outputs, label)
            loss = bce_weight * bce_loss + iou_weight * iou_loss
            # Log train loss to Comet.ml
            experiment.log_metric("train_loss", loss.item(), step=epoch * len(train_dataloader) + i)

            # Backward pass and optimization
            optimizer_rest.zero_grad()
            optimizer_upscale.zero_grad()
            loss.backward()
            if epoch < 40:
                optimizer_rest.step()
            optimizer_upscale.step()

            running_loss += loss.item()

        msg = f'Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}, Average Loss: {running_loss / len(train_dataloader)}'
        print(msg)
        experiment.log_metric("epoch_loss", running_loss, step=epoch)
        running_loss = 0.0

        # Save initial weights
        if epoch == 0:
            torch.save(model.swin.state_dict(), initial_weights_name)
            experiment.log_asset(initial_weights_name)

        if epoch % log_custom_info_at_each_nth_epoch == 0 and epoch != 0:
            torch.save(model, 'model/just_a_tranformer.pt')

            send_message(msg)
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

                    outputs, intermediate = model(image)

                    # Apply a threshold of 0.5: above -> 1, below -> 0
                    preds = outputs # (outputs > 0.15).float()
                    inter = intermediate
                    # print(torch.nonzero(preds))
                    np_preds = np.squeeze(preds.cpu().numpy())
                    np_label = np.squeeze(label.cpu().numpy())
                    np_image = np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0))
                    np_inter = np.squeeze(inter.cpu().numpy())
                    # print(np_image.shape)

                    # preds = preds.view(-1)

                    # y_true.extend(label.cpu().numpy())
                    # y_pred.extend(preds.cpu().numpy())
                    # print(label.shape, preds.shape)

                    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

                    # Display np_image1
                    axs[0].imshow(np_image, cmap='gray')
                    axs[0].axis('off')

                    # Display np_image2
                    axs[1].imshow(np_label, cmap='gray')
                    axs[1].axis('off')

                    # Display np_image3
                    axs[3].imshow(np_preds, cmap='gray')
                    axs[3].axis('off')

                    # Display np_image4
                    axs[2].imshow(np_inter, cmap='gray')
                    axs[2].axis('off')
                    # Save the figure with both subplots
                    plt.subplots_adjust(wspace=0, hspace=0)
                    plt.tight_layout()
                    plt.savefig(f'preds/combined_{i}_epoch_{epoch}.png', bbox_inches='tight', pad_inches=0)
                    # Log to Comet
                    experiment.log_figure(f'preds/combined_{i}_epoch_{epoch}.png', plt)

                    # Send an image
                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    send_photo(buf)
                    buf.close()
                    plt.close()

    torch.save(model, 'model/just_a_tranformer.pt')
    log_model(experiment, model, model_name='model/just_a_tranformer.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid')
    args = parser.parse_args()
    
    main(vars(args))
