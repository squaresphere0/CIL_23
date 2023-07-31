from dataloader import LazyImageDataset
from pixel_cnn import conditionalPixelCNN

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader


def visualize_images(num, original, mask, reconstructed):
    fig, axes = plt.subplots(nrows=3, ncols=num, figsize=(12, 6))
    for i in range(num):
        axes[0, i].imshow(original[i].squeeze().permute(1,2,0))
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")

        axes[1, i].imshow(mask[i].detach().numpy().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("True")

        axes[2, i].imshow(reconstructed[i].detach().numpy().squeeze(), cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()

def test_dataloader():
    deepglobe = LazyImageDataset(
        'Datasets/ethz-cil-road-segmentation-2023/metadata.csv',
         size=(100,100))

    DGloader = DataLoader(deepglobe, 4, False)

    for image, mask in DGloader:
        print(image.shape)
        print(mask.shape)
        visualize_images(4, image, mask)
        break

def find_mask_mean():
    data =LazyImageDataset(
        'Datasets/ethz-cil-road-segmentation-2023/metadata.csv',
        size=(400,400), rrc_scale=(1,1), rrc_ratio=(1,1))

    running_mean = 0
    count = 0
    for _, mask in data:
        running_mean += mask.mean()
        count += 1

    return running_mean / count
    # Result: 0.1780 (mask shifted: -0.6440)

def test_model(model_name, model_skeleton):
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    BATCHSIZE = 4

    original_dataset = LazyImageDataset(
        'Datasets/ethz-cil-road-segmentation-2023/metadata.csv',
        size = (100,100))

    loader = DataLoader(original_dataset, BATCHSIZE, shuffle=True)

    model = model_skeleton.to(device)

    medium_noise_model = torch.load('model/'+model_name+'.pt',
                                    map_location=device)

    model.load_state_dict(medium_noise_model['model_state_dict'])


    with torch.no_grad():
        model.train(False)
        for image, mask in loader:
            initial_guess = torch.randn(mask.shape)
            prediction = model.inference_by_iterative_refinement(100,BATCHSIZE,
                                                                10, image,
                                                                 initial_guess)
            visualize_images(BATCHSIZE, 
                             image,
                             mask,
                             prediction)


layers = [7] + [3 for _ in range(15)]

model = conditionalPixelCNN(20,1,4, layers, noise=0.5,
                            dropout_type='full')
test_model('dropout2d_08', model)
