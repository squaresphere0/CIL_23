import torch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

from just_a_transformer import PixelSwinT, ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms

import dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load your saved model
model_filename = 'just_a_tranformer_epoch_50__tuesday_unet_1st_try.pt'
print(model_filename)
model = torch.load(f'model/{model_filename}', map_location=torch.device('cpu'))

train_dataset = ImageDataset('data/training', 'cuda' if torch.cuda.is_available() else 'cpu')
val_dataset = ImageDataset('data/validation', 'cuda' if torch.cuda.is_available() else 'cpu')
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

y_true = []
y_pred = []

with torch.no_grad():
    sum_f1 = 0
    for i, (image, label) in enumerate(val_dataloader):
        print(f'Iteration: {i}')
        # resize = transforms.Resize((224, 224))

        image = image.to(device)
        label = label.to(device)

        # image = image.to(device)
        # label = label.view(-1).to(device)

        outputs, _ = model(image)

        # Apply a threshold of 0.5: above -> 1, below -> 0
        threshold = 0.25
        preds = outputs # (outputs > 0.25).float()
        # print(label.shape, preds.shape)
        sum_f1 += f1_score(label.view(-1).cpu().numpy(), (outputs > threshold).view(-1).cpu().numpy(), average='binary')  # binary case

        # print(torch.nonzero(preds))
        np_threshold = np.squeeze((outputs > threshold).float().numpy())
        np_preds = np.squeeze(preds.numpy())
        np_label = np.squeeze(label.numpy())
        np_image = np.transpose(np.squeeze(image.numpy()), (1, 2, 0))
        # print(np_image.shape)

        # preds = preds.view(-1)

        # y_true.extend(label.cpu().numpy())
        # y_pred.extend(preds.cpu().numpy())
        # print(label.shape, preds.shape)
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        # Display np_image1
        axs[0].imshow(np_image, cmap='gray')
        axs[0].axis('off')

        # Display np_image2
        axs[1].imshow(np_label, cmap='gray')
        axs[1].axis('off')

        # Display np_image3
        axs[2].imshow(np_threshold, cmap='gray')
        axs[2].axis('off')

        # Display np_image4
        axs[3].imshow(np_preds, cmap='gray')
        axs[3].axis('off')

        # Save the figure with both subplots
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.savefig(f'preds/_manual_validation_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f'Avg F1 score: {sum_f1 / len(val_dataloader)}')



# Compute F1 score