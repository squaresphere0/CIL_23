import torch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

from transformer_with_unet import ViTUNet, UnetDecoder, ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms

import dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load your saved model
model = torch.load('model/almost_my_transformer.pt', map_location=torch.device('cpu'))

train_dataset = ImageDataset('data/training', 'cuda' if torch.cuda.is_available() else 'cpu')
val_dataset = ImageDataset('data/validation', 'cuda' if torch.cuda.is_available() else 'cpu')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

y_true = []
y_pred = []

with torch.no_grad():
    for i, (image, label) in enumerate(val_dataloader):
        resize = transforms.Resize((224, 224))

        image = resize(image).to(device)
        label = resize(label).to(device)

        # image = image.to(device)
        # label = label.view(-1).to(device)

        outputs = model(image)

        # Apply a threshold of 0.5: above -> 1, below -> 0
        preds = outputs #(outputs > 0.5).float()
        print(torch.nonzero(preds))
        np_preds = np.squeeze(preds.numpy())
        np_image = np.transpose(np.squeeze(image.numpy()), (1, 2, 0))

        preds = preds.view(-1)

        y_true.extend(label.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        print(label.shape, preds.shape)
        
        plt.imshow(np_preds, cmap='gray')  # Choose appropriate colormap if your image is not grayscale
        plt.axis('off')  # To not display axes
        plt.savefig(f'preds/preds_{i}.png', bbox_inches='tight', pad_inches=0)

        plt.imshow(np_image)  # Choose appropriate colormap if your image is not grayscale
        plt.axis('off')  # To not display axes
        plt.savefig(f'preds/origs_{i}.png', bbox_inches='tight', pad_inches=0)

# Compute F1 score
f1 = f1_score(y_true, y_pred, average='binary')  # binary case
print(f"F1 Score: {f1}")