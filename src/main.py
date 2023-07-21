import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

import pixel_cnn
from pixel_cnn import conditionalPixelCNN
from pixel_cnn import visualize_images
from pixel_cnn import shift_mask
import dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu' 


original_dataset = dataloader.LazyImageDataset(
    'Datasets/ethz-cil-road-segmentation-2023/metadata.csv',
    size = (100,100))

loader = DataLoader(original_dataset, 4, shuffle=True)

layers = [7] + [3 for _ in range(15)]

model = conditionalPixelCNN(20,1,4, layers, noise=0.5).to(device)

optimizer = torch.optim.Adam(model.parameters())

medium_noise_model = torch.load('model/non_auto_regressive.pt',
                                map_location=device)
model.load_state_dict(medium_noise_model['model_state_dict'])


losses = conditionalPixelCNN.training(model,loader,optimizer, 10,
                                      'finetuned_10epochs')

