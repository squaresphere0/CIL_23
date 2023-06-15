import torch
from torch import nn
import patch_cnn
import image_dataset
import training_loop
from pathlib import Path

BASE_PATH = Path('.')


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # automatically select device
train_dataset = image_dataset.ImageDataset(BASE_PATH / 'data' / 'training',
                                           device)
val_dataset = image_dataset.ImageDataset(BASE_PATH / 'data' / 'validation', device)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)
model = patch_cnn.PatchCNN().to(device)
loss_fn = nn.BCELoss()
metric_fns = {'acc': patch_cnn.accuracy_fn}
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 20
training_loop.train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)
