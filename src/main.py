import comet_ml

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import torchvision
from torchview import draw_graph

import matplotlib.pyplot as plt

import pixel_cnn
from pixel_cnn import conditionalPixelCNN
from pixel_cnn import visualize_images
from pixel_cnn import shift_mask
import dataloader


# Initialize logging to Comet
comet_experiment = comet_ml.Experiment(
    api_key = "x6UJjWwiy9x4Z3RaBjZ4hEHGk",
    project_name = "cil-23",
    workspace="mrpetrkol"
)

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

BATCHSIZE = 4

original_dataset = dataloader.LazyImageDataset(
    'Datasets/ethz-cil-road-segmentation-2023/metadata.csv',
    size = (100,100))

loader = DataLoader(original_dataset, BATCHSIZE, shuffle=True)

layers = [7] + [3 for _ in range(15)]

model = conditionalPixelCNN(20,1,4, layers, noise=0.8).to(device)

optimizer = torch.optim.Adam(model.parameters())

medium_noise_model = torch.load('model/finetuned_2epochs.pt',
                                map_location=device)
model.load_state_dict(medium_noise_model['model_state_dict'])

down_sample = nn.AvgPool2d(4)

# with torch.no_grad():
#     model.eval()
#     for image, mask in loader:
#         bias_func = lambda a : 0.8 * ( a - 0.0)
#         pred = model.inference_by_iterative_refinement(bias_func, 10, BATCHSIZE, 100, image)
#         #image.movedim(1,3)
#         visualize_images(mask, pred)

model_graph = draw_graph(model, input_size=(BATCHSIZE, 3, 100, 100), expand_nested=True)

losses = conditionalPixelCNN.training(model,loader,optimizer, 200,
                                      '08_noise_200epochs', noise=0.1, experiment=comet_experiment)

