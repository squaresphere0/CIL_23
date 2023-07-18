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

def shift_mask(mask):
    return 2 * (mask - 0.5)

def train(model, loader, optimizer, epochs):
    model.train()
    losses = []
    for epoch in range(epochs):
        for i, (image, mask) in enumerate(loader):
            image = image.to(device)
            mask = mask.to(device)
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

    return losses



transform = transforms.Compose([transforms.RandomResizedCrop((100,100)),
                                 transforms.ToTensor()])

original_dataset = dataloader.LazyImageDataset(
    'Datasets/ethz-cil-road-segmentation-2023/metadata.csv',
    transform)

loader = DataLoader(original_dataset, 32, shuffle=True)

model = conditionalPixelCNN(20,1,4, (7,5,5,3,3,3))

optimizer = torch.optim.Adam(model.parameters())

losses = train(model,loader,optimizer,200)

torch.save({'model_state_dict': model.state_dict(),
            'loss_history': losses
           }, 'model/7_9_cond.pt')
