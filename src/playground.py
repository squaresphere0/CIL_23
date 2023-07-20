from dataloader import LazyImageDataset

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def visualize_images(num, original, reconstructed):
    fig, axes = plt.subplots(nrows=2, ncols=num, figsize=(12, 6))
    for i in range(num):
        axes[0, i].imshow(original[i].squeeze().permute(1,2,0))
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")

        axes[1, i].imshow(reconstructed[i].detach().numpy().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()

deepglobe = LazyImageDataset(
    'Datasets/ethz-cil-road-segmentation-2023/metadata.csv',
     size=(100,100))

DGloader = DataLoader(deepglobe, 4, False)

for image, mask in DGloader:
    print(image.shape)
    print(mask.shape)
    visualize_images(4, image, mask)
    break

import torch
from torch import nn
m = nn.Dropout(p=1)
in_tens = torch.randn(20, 16)
output = m(in_tens)
print(in_tens)
print(output)
