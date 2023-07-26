import comet_ml
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
from pixel_cnn import shift_mask
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

model = conditionalPixelCNN(20,1,4, layers, noise=0.0).to(device)

optimizer = torch.optim.Adam(model.parameters())

'''
medium_noise_model = torch.load('model/gaussian_noise_01.pt',
                                map_location=device)

model.load_state_dict(medium_noise_model['model_state_dict'])


avg = []
with torch.no_grad():
    model.train(False)
    for image, mask in loader:
        bias = lambda a: a * 0.0
        prediction = model.inference_by_iterative_refinement(bias,1,BATCHSIZE,
                                                             100, image)
        prediction = model(torch.cat((image, mask),1))
        visualize_images(image.movedim(1,3),
                         prediction)
        break

print(avg)

'''

# # Plot model graph.
# model_graph = draw_graph(model, input_size=(BATCHSIZE, 3, 100, 100), expand_nested=True)
# model_graph_json = model_graph.visual_graph.render(filename='temp_graph', format='svg', cleanup=True)
# cairosvg.svg2png(url='temp_graph.svg', write_to='temp_graph.png')
# with open('temp_graph.png', 'rb') as f:
#     image_bytes = f.read()
#     comet_experiment.log_asset_data(image_bytes, name='graph.png', overwrite=True)

losses = conditionalPixelCNN.training(model,loader,optimizer, 200,
                                      'gaussian_noise_03', noise=0.3, experiment=comet_experiment)
