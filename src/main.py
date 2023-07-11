import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import pixel_cnn
from pixel_cnn import conditionalPixelCNN
from pixel_cnn import visualize_images
import dataloader

def shift_mask(mask):
    return 2 * (mask - 0.5)

def train(model, loader, optimizer, epochs):
    model.train()
    losses = []
    for epoch in range(epochs):
        for i, (image, mask) in enumerate(loader):

            generated = model(torch.cat(
                (shift_mask(mask), image), 1))

            loss_function = nn.BCELoss()
            loss = loss_function(generated, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:

                print("Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, epochs, i + 1, len(loader), loss.item()
                ))
                #visualize_images(mask, generated)
            losses.append(loss.detach())

    torch.save({'model_state_dict': model.state_dict(),
                'loss_history': losses
               }, 'model/7_9_cond.pt')



original_dataset = dataloader.LazyImageDataset(
    'Datasets/ethz-cil-road-segmentation-2023/metadata.csv')
loader = DataLoader(original_dataset, 32, shuffle=True)

model = conditionalPixelCNN(20,1,4)

optimizer = torch.optim.Adam(model.parameters())

train(model,loader,optimizer,20)
