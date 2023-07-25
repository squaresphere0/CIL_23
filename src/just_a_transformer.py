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
import copy
from sklearn.metrics import f1_score

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import torchvision
import torchview

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

from transformer_with_unet import ViTUNet, UnetDecoder
import torch
from torch import nn
import timm


import torch
from torch import nn
import timm
from torchvision.transforms import Resize
from torchvision import transforms
# from efficientnet_pytorch import EfficientNet


class PixelSwinT(nn.Module):
    def __init__(self, swin_model_name='swinv2_large_window12to24_192to384'):
        super().__init__()

        self.switch_to_simultaneous_training_after_epochs = 30

        self.current_epoch = 0


        # Load the SWIN Transformer model, but remove the classification head
        self.swin = timm.create_model(swin_model_name, pretrained=True, num_classes=0)
        self.swin.head = nn.Identity()

        self.resize = Resize((384, 384))

        # self.dropout = nn.Dropout(p=0.5)

        num_channels = 1536
        self.reduce_channels = nn.Conv2d(num_channels, 1, kernel_size=1)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1536, out_channels=768, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(768),
            nn.ReLU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1536, out_channels=768, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(768),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1152, out_channels=576, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(),
        )
        self.not_up3 = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(768),
            nn.ReLU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=768, out_channels=384, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=384, out_channels=1, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        # self.not_up6 = nn.Sequential(
        #     nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=2, dilation=2),
        #     nn.BatchNorm2d(768),
        #     nn.ReLU(),
        # )

        # self.upscale = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=4, stride=2, padding=1, output_padding=0),
        #     nn.BatchNorm2d(num_channels // 2),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(in_channels=num_channels // 2, out_channels=num_channels // 4, kernel_size=4, stride=2, padding=1, output_padding=0),
        #     nn.BatchNorm2d(num_channels // 4),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(in_channels=num_channels // 4, out_channels=num_channels // 8, kernel_size=4, stride=2, padding=1, output_padding=0),
        #     nn.BatchNorm2d(num_channels // 8),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(in_channels=num_channels // 8, out_channels=num_channels // 16, kernel_size=4, stride=2, padding=1, output_padding=0),
        #     nn.BatchNorm2d(num_channels // 16),
        #     nn.ReLU(),

        #     nn.ConvTranspose2d(in_channels=num_channels // 16, out_channels=num_channels // 32, kernel_size=4, stride=2, padding=1, output_padding=0),
        #     nn.BatchNorm2d(num_channels // 32),
        #     nn.ReLU(),

        #     nn.Conv2d(in_channels=num_channels // 32, out_channels=1, kernel_size=1),  # Output layer, now with 1 channel
        # )
        self.upsample = nn.Upsample(size=(400, 400), mode='bicubic') #, align_corners=True)
        self.batchnorm = nn.Sequential(
            # nn.Conv2d(1536, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        # stage1 = self.swin.layers[0](x)
        # print(f'stage1.shape: {stage1.shape}')


        x = self.resize(x)

        embed = self.swin.patch_embed(x)
        stage0 = self.swin.layers[0](embed)
        stage1 = self.swin.layers[1](stage0)
        stage2 = self.swin.layers[2](stage1)
        stage3 = self.swin.layers[3](stage2)

        up0 = self.up0(stage3.permute(0, 3, 1, 2))        
        up1 = self.up1(torch.cat([up0, stage2.permute(0, 3, 1, 2)], dim=1))
        up2 = self.up2(torch.cat([up1, stage1.permute(0, 3, 1, 2)], dim=1))
        not_up3 = self.not_up3(torch.cat([up2, stage0.permute(0, 3, 1, 2)], dim=1))
        up4 = self.up4(not_up3)
        up5 = self.up5(up4)


        swin_x = self.swin(x)
        swin_x = swin_x.permute(0, 3, 1, 2)  # permute the dimensions to bring it to (B, Channels, H, W) format
        intermediate = self.reduce_channels(swin_x)
        intermediate = self.upsample(intermediate)
        # intermediate = self.classifier(intermediate)
        # x = self.reduce_dim(x)  # reduce dimensionality to 1
        # print(x.shape)
        # x = F.interpolate(x, size=(224, 224))
        # print("SHape after swin:", x.shape)

        if self.current_epoch <= self.switch_to_simultaneous_training_after_epochs:
            x = self.upsample(swin_x)
            x = self.reduce_channels(x)
            x = self.batchnorm(x)
            return x, intermediate

        x = self.upsample(up5)  # Upsample to the original image size
        x = self.batchnorm(x)
        # x = self.classifier(x)  # Classify each pixel
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


class RotationTransform:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = int(np.random.choice(self.angles))
        # Rotate and convert tensor to PIL for rotation
        rotated_PIL = transforms.functional.rotate(transforms.ToPILImage()(x), angle)
        return transforms.ToTensor()(rotated_PIL)


class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, path, device, use_patches=False, resize_to=(400, 400)):
        self.path = path
        self.is_train = 'train' in path
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
        if not self.is_train:
            return x, y
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        transform = RotationTransform([0, 90, 180, 270])
        x = transform(x)
        y = transform(y)  # Apply the same transformation to y
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


def send_photo(photo, caption_text=''):
    url = "https://api.telegram.org/bot6519873169:AAGxxszlbXMh9CQg9L4gK4EIOGVfcOZE2RI/sendPhoto"
    files = {'photo': photo}
    data = {
        'chat_id': "502129529",
        'disable_notification': True,
        'caption': caption_text,  # Add your caption here
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


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        intersection = (preds * targets).sum()
        dice_coeff = (2.*intersection + self.smooth)/(preds.sum() + targets.sum() + self.smooth)
        return 1. - dice_coeff


def main(args):
    # send_message("Loaded to the execution environment.")

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

    log_custom_info_at_each_nth_epoch = 10

    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiment = Experiment(
        api_key = "x6UJjWwiy9x4Z3RaBjZ4hEHGk",
        project_name = "cil-23",
        workspace="mrpetrkol"
    )

    # Create the model and move it to the GPU if available
    model = PixelSwinT().to(device)

    initial_weights_name = f'model/{experiment.get_name()}_initial_swin_weights.pth'
    initial_weights = model.swin.state_dict()
    torch.save(initial_weights, initial_weights_name)
    # if os.path.isfile(initial_weights_name):
    #     model.swin.load_state_dict(torch.load(initial_weights_name))

    # Specify a loss function and an optimizer
    metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}

    bce_loss_pos_weight = 20.0
    bce_loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_loss_pos_weight]).to(device))
    bce_loss_function_after_n_epochs = nn.BCEWithLogitsLoss()
    extra_loss_function = DiceLoss()

    bce_weight = 1  # This determines how much the BCE loss contributes to the total loss
    extra_weight = 1 - bce_weight  # This determines how much the IoU loss contributes to the total loss        optimizer = torch.optim.Adam(model.parameters())

    optimizer = torch.optim.Adam(model.parameters())
    # rest_of_model_params = [p for n, p in model.named_parameters() if 'upscale' not in n]
    # optimizer_rest = torch.optim.Adam(rest_of_model_params)
    # optimizer_upscale = torch.optim.Adam(model.upscale.parameters(), weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    my_batch_size = 2
    train_dataset = ImageDataset('data/training', 'cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset = ImageDataset('data/validation', 'cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=my_batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

    num_epochs = 300

    hyper_params = {
        # "learning_rate": optimizer.param_groups[0]['lr'],
        # "weight_decay": optimizer.param_groups[0]['weight_decay'],
        "num_epochs": num_epochs,
        "batch_size": my_batch_size,
        'bce_weight': bce_weight,
        'extra_weight': extra_weight,
        'bce_loss_pos_weight': bce_loss_pos_weight,
        'switch_to_simultaneous_training_after_epochs': model.switch_to_simultaneous_training_after_epochs,
    }
    experiment.log_parameters(hyper_params)
    experiment.set_model_graph(str(model))

    # Visualize the model
    torchview.draw_graph(model, input_size=(my_batch_size, 3, 400, 400), depth=1)#, expand_nested=True)
    cairosvg.svg2png(url='temp_graph.svg', write_to='temp_graph.png')
    with open('temp_graph.png', 'rb') as f:
        image_bytes = f.read()
        experiment.log_asset_data(image_bytes, name='graph.png', overwrite=True)

    send_message("Starting new computation.")
    msg = "First epoch."
    for epoch in range(num_epochs):
        if epoch % log_custom_info_at_each_nth_epoch == 0 or epoch == num_epochs - 1:
            send_message(msg)
            # Evaluate on the validation set
            print("Evaluating, plotting images.")
            model.eval()  # Put the model in evaluation mode
            with torch.no_grad():
                sum_f1 = 0
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

                    sum_f1 += f1_score(label.view(-1).cpu().numpy(), (outputs >= 0.25).float().view(-1).cpu().numpy(), average='binary')  # binary case
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
                    send_photo(buf, f1_score(label.view(-1).cpu().numpy(), (outputs >= 0.25).float().view(-1).cpu().numpy(), average='binary'))  # binary case)
                    buf.close()
                    plt.close()

            print(f'Avg F1 score: {sum_f1 / len(val_dataloader)}')
            send_message(f'Avg F1 score: {sum_f1 / len(val_dataloader)}')
            experiment.log_metric("avg_f1_score", sum_f1 / len(val_dataloader), step=epoch)


        model.train()  # Put the model in training mode
        running_loss = 0.0
        step_counter = 0
        model.current_epoch = epoch
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
            if epoch > model.switch_to_simultaneous_training_after_epochs:
                bce_loss = bce_loss_function_after_n_epochs(outputs, label)
            extra_loss = extra_loss_function(outputs, label)
            loss = bce_weight * bce_loss + extra_weight * extra_loss
            # Log train loss to Comet.ml
            experiment.log_metric("train_loss", loss.item(), step=epoch * len(train_dataloader) + i)

            # Backward pass and optimization
            optimizer.zero_grad()
            # optimizer_upscale.zero_grad()
            loss.backward()
            # if epoch < 40:
            optimizer.step()
            # optimizer_upscale.step()

            running_loss += loss.item()
            step_counter += 1

        msg = f'Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}, Average Loss: {running_loss / step_counter}'
        print(msg)
        experiment.log_metric("epoch_loss", running_loss / step_counter, step=epoch)
        running_loss = 0.0

        if epoch % 50 == 0 and epoch != 0:
            torch.save(model, f'model/{experiment.get_name()}_just_a_tranformer_epoch_{epoch}.pt')
        if epoch == 50:
            experiment.log_asset(initial_weights_name)

    model_name = f'model/{experiment.get_name()}_just_a_tranformer_epoch_{num_epochs}.pt'
    torch.save(model, model_name)
    experiment.log_asset(model_name)
    # log_model(experiment, model, model_name)
    # experiment.log_asset(initial_weights_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid')
    args = parser.parse_args()
    
    main(vars(args))
