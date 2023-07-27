import os

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


MODEL_NAME = 'developing_cinema_6230_just_a_tranformer_epoch_210'  # without .pt extension


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

BATCHSIZE = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

original_dataset = dataloader.LazyImageDataset('Datasets/ethz-cil-road-segmentation-2023/metadata.csv')

loader = DataLoader(original_dataset, BATCHSIZE, shuffle=True)

layers = [7] + [3 for _ in range(15)]



#optimizer = torch.optim.Adam(model.parameters())
#medium_noise_model = torch.load('model/non_auto_regressive_200epochs_0.8noise.pt',
#                                map_location=device)
#model.load_state_dict(medium_noise_model['model_state_dict'])
model = torch.load(f'model/{MODEL_NAME}.pt', map_location=torch.device('cpu'))




    


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


image_paths = ["satimage_250.png",
"satimage_244.png",
"satimage_278.png",
"satimage_287.png",
"satimage_286.png",
"satimage_279.png",
"satimage_245.png",
"satimage_251.png",
"satimage_247.png",
"satimage_253.png",
"satimage_284.png",
"satimage_285.png",
"satimage_252.png",
"satimage_246.png",
"satimage_242.png",
"satimage_256.png",
"satimage_281.png",
"satimage_280.png",
"satimage_257.png",
"satimage_243.png",
"satimage_269.png",
"satimage_255.png",
"satimage_241.png",
"satimage_282.png",
"satimage_283.png",
"satimage_240.png",
"satimage_254.png",
"satimage_268.png",
"satimage_145.png",
"satimage_151.png",
"satimage_179.png",
"satimage_186.png",
"satimage_192.png",
"satimage_233.png",
"satimage_227.png",
"satimage_226.png",
"satimage_232.png",
"satimage_193.png",
"satimage_187.png",
"satimage_178.png",
"satimage_150.png",
"satimage_144.png",
"satimage_152.png",
"satimage_146.png",
"satimage_191.png",
"satimage_185.png",
"satimage_224.png",
"satimage_230.png",
"satimage_218.png",
"satimage_219.png",
"satimage_231.png",
"satimage_225.png",
"satimage_184.png",
"satimage_190.png",
"satimage_147.png",
"satimage_153.png",
"satimage_157.png",
"satimage_194.png",
"satimage_180.png",
"satimage_209.png",
"satimage_221.png",
"satimage_235.png",
"satimage_234.png",
"satimage_220.png",
"satimage_208.png",
"satimage_181.png",
"satimage_195.png",
"satimage_156.png",
"satimage_168.png",
"satimage_154.png",
"satimage_183.png",
"satimage_197.png",
"satimage_236.png",
"satimage_222.png",
"satimage_223.png",
"satimage_237.png",
"satimage_196.png",
"satimage_182.png",
"satimage_155.png",
"satimage_169.png",
"satimage_164.png",
"satimage_170.png",
"satimage_158.png",
"satimage_212.png",
"satimage_206.png",
"satimage_207.png",
"satimage_213.png",
"satimage_159.png",
"satimage_171.png",
"satimage_165.png",
"satimage_173.png",
"satimage_167.png",
"satimage_198.png",
"satimage_205.png",
"satimage_211.png",
"satimage_239.png",
"satimage_238.png",
"satimage_210.png",
"satimage_204.png",
"satimage_199.png",
"satimage_166.png",
"satimage_172.png",
"satimage_176.png",
"satimage_162.png",
"satimage_189.png",
"satimage_228.png",
"satimage_200.png",
"satimage_214.png",
"satimage_215.png",
"satimage_201.png",
"satimage_229.png",
"satimage_188.png",
"satimage_163.png",
"satimage_177.png",
"satimage_149.png",
"satimage_161.png",
"satimage_175.png",
"satimage_217.png",
"satimage_203.png",
"satimage_202.png",
"satimage_216.png",
"satimage_174.png",
"satimage_160.png",
"satimage_148.png",
"satimage_271.png",
"satimage_265.png",
"satimage_259.png",
"satimage_258.png",
"satimage_264.png",
"satimage_270.png",
"satimage_266.png",
"satimage_272.png",
"satimage_273.png",
"satimage_267.png",
"satimage_263.png",
"satimage_277.png",
"satimage_276.png",
"satimage_262.png",
"satimage_248.png",
"satimage_274.png",
"satimage_260.png",
"satimage_261.png",
"satimage_275.png",
"satimage_249.png"]
image_paths = sorted(image_paths)
mask_paths = [f"Datasets/ethz-cil-road-segmentation-2023/test/pred_{MODEL_NAME}/" + i for i in image_paths]
image_paths = ["Datasets/ethz-cil-road-segmentation-2023/test/images/" + i for i in image_paths]

resized_tensors = resize_images(image_paths)




with torch.no_grad():
    model.eval()
    for i in range(0, len(resized_tensors)):
        print(f'Iteration: {i}')
        tensor = resized_tensors[i].unsqueeze(0)
        tensor = tensor.to(device)


        # np_preds = np.transpose(np.squeeze(tensor.cpu().numpy()), (1, 2, 0))
        # fig, axs = plt.subplots(2, 1, figsize=(5, 5))
        # # Display np_image
        # axs[0].imshow(np_preds, cmap='gray')
        # axs[0].axis('off')
        # plt.subplots_adjust(wspace=0, hspace=0)
        # plt.tight_layout()
        
        # plt.show()
        
        generated, inter = model(tensor)
        generated = generated


        j = 0
        for g in generated:
            # Convert the generated mask tensor to a NumPy array and then to integers
            generated_mask = g
            resized_mask = generated_mask
            # Resize the NumPy array to 400x400 using scikit-image's resize function
            # resized_mask = skimage.transform.resize(generated_mask, (400, 400), order=3, anti_aliasing=True)
            # Scale the floating-point values to [0, 255] and convert to integers
            # resized_mask = (resized_mask * 255).astype(np.int)
            # Save the resized mask as an image
            save_mask_as_img(resized_mask, mask_paths[i + j])
            j = j + 1
