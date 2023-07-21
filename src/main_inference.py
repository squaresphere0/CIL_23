import torch
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import pixel_cnn
from pixel_cnn import conditionalPixelCNN
from pixel_cnn import visualize_images
import dataloader

i = 0
generated = [0,0,0,0,0]
for images, masks in loader:
    if i > 1 :
        break
    # plt.imshow(images[i].permute(1, 2, 0))
    # plt.axis('off')
    # plt.show()

    # plt.imshow(masks[i].permute(1, 2, 0))
    # plt.axis('off')
    # plt.show()

    generated[i] = model.generate_samples(1, 256, images[i].unsqueeze(0).to(torch.device('cuda')))


    # output =
    # plt.imshow(output.cpu().squeeze(0).permute(1,2,0))
    plt.savefig(f'output_{i}.png', bbox_inches='tight', pad_inches = 0)
    plt.axis('off')
    # plt.show()
    i = i + 1
# visualize_images(masks[0:2].cpu(), generated[0:2])