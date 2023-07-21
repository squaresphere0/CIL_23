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

medium_noise_model = torch.load('model/medium_noise_model.pt',
                                map_location=device)
model.load_state_dict(medium_noise_model['model_state_dict'])


for images, masks in loader:
    model.eval()
    empty_mask = torch.randn(4,1,100,100)
    for i in range(50):
        if i % 10 == 0: 
            visualize_images(masks, empty_mask)
        generated = model(torch.cat((empty_mask, images), 1))
        empty_mask = (generated  -0.3)*0.5
    break
'''
losses = conditionalpixelcnn.training(model,loader,optimizer, 2000,
                                      'noise_0.5_epoch_2000')
                                      '''

