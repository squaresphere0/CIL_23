import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

import pixel_cnn
from pixel_cnn import conditionalPixelCNN
from pixel_cnn import visualize_images
import dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu' 


original_dataset = dataloader.LazyImageDataset(
    'Datasets/ethz-cil-road-segmentation-2023/metadata.csv',
    size = (100,100))

loader = DataLoader(original_dataset, 4, shuffle=True)

layers = [7] + [3 for _ in range(15)]

model = conditionalPixelCNN(20,1,4, layers, noise=0.5).to(device)

optimizer = torch.optim.Adam(model.parameters())

medium_noise_model = torch.load('model/medium_noise_model.pt',
                                map_location=device)
losses = medium_noise_model['loss_history']
model.load_state_dict(medium_noise_model['model_state_dict'])


for images, masks in loader:
    model.eval()
    generated = model.generate_samples(4, 100, images)
    visualize_images(masks, generated)
'''
losses = conditionalPixelCNN.training(model,loader,optimizer, 2000,
                                      'noise_0.5_epoch_2000')
                                      '''

