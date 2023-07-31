from mask_to_submission import masks_to_submission, save_mask_as_img
from torchvision.transforms import ToTensor, Resize
from PIL import Image

import skimage.transform

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

import pixel_cnn
from pixel_cnn import conditionalPixelCNN
import dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def resize_images(image_paths, size):
    resized_images = []
    resize_transform = Resize((size, size))
    to_tensor_transform = ToTensor()

    for path in image_paths:
        image = Image.open(path)
        resized_image = resize_transform(image)
        tensor_image = to_tensor_transform(resized_image)
        resized_images.append(tensor_image)

    return resized_images

def create_masks(model_name, model_skeleton):

    image_size = 100
    BATCHSIZE = 16

    image_paths = ["satimage_"+str(i)+".png" for i in range(144,288,1)]
    mask_paths = ["Datasets/ethz-cil-road-segmentation-2023/test/" + model_name
                  + "/" + i for i in image_paths]
    image_paths = ["Datasets/ethz-cil-road-segmentation-2023/test/images/" 
                   + i for i in image_paths]

    resized_tensors = resize_images(image_paths, image_size)

    model = model_skeleton.to(device)
    model.load_state_dict(torch.load('model/'+model_name+'.pt',
                                     map_location=device)['model_state_dict'])

    with torch.no_grad():
        model.eval()
        for i in range(0, len(resized_tensors), BATCHSIZE):
            batch = resized_tensors[i:i + BATCHSIZE]
            batch_tensor = torch.stack(batch, dim=0)

            print(batch_tensor.shape)

            initial_guess = torch.randn(BATCHSIZE, 1, image_size, image_size)
            generated = model.inference_by_iterative_refinement(
                100,
                BATCHSIZE,
                image_size,
                batch_tensor,
                initial_guess)

            j = 0
            for g in generated:
                # Convert the generated mask tensor to a NumPy array and then to integers
                generated_mask = (g.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
                # Resize the NumPy array to 400x400 using scikit-image's resize function
                resized_mask = skimage.transform.resize(generated_mask, (400, 400), order=3, anti_aliasing=True)
                # Scale the floating-point values to [0, 255] and convert to integers
                resized_mask = (resized_mask * 255).astype(np.uint8)
                # Save the resized mask as an image
                save_mask_as_img(resized_mask, mask_paths[i + j])
                j = j + 1


layers = [7] + [3 for _ in range(15)]

model = conditionalPixelCNN(20,1,4, layers, noise=0.8,
                            dropout_type='full')
create_masks('dropout2d_08', model)
