import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

import pixel_cnn
from pixel_cnn import conditionalPixelCNN
from pixel_cnn import visualize_images
import dataloader

def visualize_images(original, reconstructed):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    for i in range(4):
        axes[0, i].imshow(original[i].permute(1, 2, 0).squeeze())
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")
        axes[1, i].imshow(reconstructed[i].detach().numpy().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

BATCHSIZE = 32

original_dataset = dataloader.LazyImageDataset(
    'Datasets/ethz-cil-road-segmentation-2023/metadata.csv',
    size = (100,100))

loader = DataLoader(original_dataset, BATCHSIZE, shuffle=True)

layers = [7] + [3 for _ in range(15)]

model = conditionalPixelCNN(20,1,4, layers, noise=0.0).to(device)

optimizer = torch.optim.Adam(model.parameters())

medium_noise_model = torch.load('model/0.8_drop.pt',
                                map_location=device)

model.load_state_dict(medium_noise_model['model_state_dict'])

down_sample = nn.AvgPool2d(4)

with torch.no_grad():
    model.eval()
    for image, mask in loader:
        bias_func = lambda a : 1 * ( a + 0.1)
        pred = model.inference_by_iterative_refinement(bias_func, 10, BATCHSIZE, 100, image)
        #image.movedim(1,3)
        visualize_images(mask, pred)
        break
'''

losses = conditionalPixelCNN.training(model,loader,optimizer, 200,
                                      'gaussian_noisei_03', noise=0.3)
'''
