import comet_ml
from comet_ml import Experiment
# import cairosvg

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

# import torchvision
# from torchview import draw_graph

import matplotlib.pyplot as plt

import pixel_cnn
from pixel_cnn import conditionalPixelCNN
from pixel_cnn import visualize_images
import dataloader


# Initialize logging to Comet

comet_experiment = Experiment(
  api_key = "zwN6QzFQ4jB78DoMN4W1ItiOo",
  project_name = "cil-23",
  workspace="squaresphere0"
)

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

BATCHSIZE = 32

original_dataset = dataloader.LazyImageDataset(
    'Datasets/ethz-cil-road-segmentation-2023/metadata.csv',
    size = (100,100))

loader = DataLoader(original_dataset, BATCHSIZE, shuffle=True)

layers = [7] + [3 for _ in range(15)]

model = conditionalPixelCNN(20,1,4, layers, noise=0.8).to(device)

optimizer = torch.optim.Adam(model.parameters())


# # Plot model graph.
# model_graph = draw_graph(model, input_size=(BATCHSIZE, 3, 100, 100), expand_nested=True)
# model_graph_json = model_graph.visual_graph.render(filename='temp_graph', format='svg', cleanup=True)
# cairosvg.svg2png(url='temp_graph.svg', write_to='temp_graph.png')
# with open('temp_graph.png', 'rb') as f:
#     image_bytes = f.read()
#     comet_experiment.log_asset_data(image_bytes, name='graph.png', overwrite=True)

losses = conditionalPixelCNN.training(model,loader,optimizer, 1000,
                                      'dropout2d_08_1000ep',
                                      noise=0.0, pos_weight=1, experiment=comet_experiment)
