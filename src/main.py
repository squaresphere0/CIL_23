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


transform = transforms.Compose([transforms.RandomResizedCrop((100,100)),
                                 transforms.ToTensor()])

original_dataset = dataloader.LazyImageDataset(
    'Datasets/ethz-cil-road-segmentation-2023/metadata.csv',
    transform)

loader = DataLoader(original_dataset, 32, shuffle=True)

model = conditionalPixelCNN(20,1,4, (7,5,5,3,3,3))

optimizer = torch.optim.Adam(model.parameters())

losses = conditionalPixelCNN.training(model,loader,optimizer,2, 'test')

