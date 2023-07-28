import os
import argparse

from mask_to_submission import masks_to_submission
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import skimage.transform


import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

from just_a_transformer import PixelSwinT
import dataloader


MODEL_NAME = 'precious_panda_1942_just_a_tranformer_epoch_80'  # without .pt extension
BATCHSIZE = 1


def save_mask_as_img(preds_tensor, mask_filename):
    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)

    np_preds = np.squeeze(preds_tensor.cpu().numpy())

    fig, axs = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    axs.imshow(np_preds, cmap='gray')
    axs.axis('off')
    # plt.tight_layout()
    plt.savefig(mask_filename, dpi=100)

    # img = PIL.Image.fromarray(img_arr)
    # img.save(mask_filename)


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


def resize_images(image_paths, size=100, resize=False):
    if not resize:
        resized_images = []
        to_tensor_transform = ToTensor()
        for path in image_paths:
            image = Image.open(path)
            tensor_image = to_tensor_transform(image)
            tensor_image = tensor_image[:3, :, :]
            # print("shape: " + str(tensor_image.shape))
            resized_images.append(tensor_image)

        return resized_images

    resized_images = []
    resize_transform = Resize((size, size))
    to_tensor_transform = ToTensor()
    for path in image_paths:
        image = Image.open(path)
        resized_image = resize_transform(image)
        tensor_image = to_tensor_transform(resized_image)
        tensor_image = tensor_image[:3, :, :] 
        # print("shape: " + str(tensor_image.shape))
        resized_images.append(tensor_image)

    return resized_images


def main(args):
    path_to_test_images_folder = 'Datasets/ethz-cil-road-segmentation-2023/test/images/'
    image_paths = sorted(os.listdir(path_to_test_images_folder))

    if args.start_from:
        image_paths = image_paths[args.start_from:args.start_from + 10]
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = MODEL_NAME  # Arguments are more important

    #optimizer = torch.optim.Adam(model.parameters())
    #medium_noise_model = torch.load('model/non_auto_regressive_200epochs_0.8noise.pt',
    #                                map_location=device)
    #model.load_state_dict(medium_noise_model['model_state_dict'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    original_dataset = dataloader.LazyImageDataset('Datasets/ethz-cil-road-segmentation-2023/metadata.csv')
    loader = DataLoader(original_dataset, BATCHSIZE, shuffle=True)

    layers = [7] + [3 for _ in range(15)]
    model = torch.load(f'model/{model_name}.pt', map_location=torch.device(device))

    mask_paths = [f"Datasets/ethz-cil-road-segmentation-2023/test/pred_{model_name}/" + i for i in image_paths]
    image_paths = ["Datasets/ethz-cil-road-segmentation-2023/test/images/" + i for i in image_paths]

    resized_tensors = resize_images(image_paths)

    with torch.no_grad():
        model.eval()
        for i in range(0, len(resized_tensors)):
            print(f'Iteration: {i}')
            tensor = resized_tensors[i].unsqueeze(0)
            tensor = tensor.to(device)

            generated, inter = model(tensor)
            generated = generated


            j = 0
            for g in generated:
                # Convert the generated mask tensor to a NumPy array and then to integers
                generated_mask = g
                resized_mask = generated_mask  # (generated_mask > 0.5).float()
                # Resize the NumPy array to 400x400 using scikit-image's resize function
                # resized_mask = skimage.transform.resize(generated_mask, (400, 400), order=3, anti_aliasing=True)
                # Scale the floating-point values to [0, 255] and convert to integers
                # resized_mask = (resized_mask * 255).astype(np.int)
                # Save the resized mask as an image
                save_mask_as_img(resized_mask, mask_paths[i + j])
                j = j + 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_from', type=int, help='From which image from 144 to 287 to start (renumbered from 0 to 143)')
    parser.add_argument('--model_name', type=str, help='From which image from 144 to 287 to start (renumbered from 0 to 143)')
    args = parser.parse_args()

    main(args)
