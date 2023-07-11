import torch
from torch import nn
import argparse
import patch_cnn
import u_net
import image_dataset
import training_loop
from pathlib import Path

BASE_PATH = Path('.')


def main(args_baseline):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # automatically select device
    if args_baseline == "patch_cnn":
        train_dataset = image_dataset.ImageDataset(BASE_PATH / 'data' / 'training', device)
        val_dataset = image_dataset.ImageDataset(BASE_PATH / 'data' / 'validation', device)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)
        model = patch_cnn.PatchCNN().to(device)
        n_epochs = 20
    elif args_baseline in ("unet", "u_net"):
        train_dataset = image_dataset.ImageDataset(BASE_PATH / 'data' / 'training', device, use_patches=False, resize_to=(384, 384))
        val_dataset = image_dataset.ImageDataset(BASE_PATH / 'data' / 'validation', device, use_patches=False, resize_to=(384, 384))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)
        model = u_net.UNet().to(device)
        n_epochs = 35


    # model = patch_cnn.PatchCNN().to(device)
    loss_fn = nn.BCELoss()
    metric_fns = {'acc': patch_cnn.accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters())
    training_loop.train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, help='Baseline to run: "patch_cnn" or "u_net"')
    args = parser.parse_args()

    main(args.baseline)
