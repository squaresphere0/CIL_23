import csv
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize

import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, RandomResizedCrop,functional

class LazyImageDataset(Dataset):
    def __init__(self, csv_file, transform = ToTensor(), size =(200,200)):
        self.csv_file = csv_file
        self.image_paths, self.mask_paths = self._read_csv()
        self.transform = transform
        self.resize = Resize(size) # Resize the images to a common size

    def _read_csv(self):
        image_paths = []
        mask_paths = []

        with open(self.csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)  # Get the header row

            if 'images' in header and 'groundtruth' in header:
                image_index = header.index('images')
                mask_index = header.index('groundtruth')
            elif 'tiff_image_path' in header and 'tif_label_path' in header:
                image_index = header.index('tiff_image_path')
                mask_index = header.index('tif_label_path')
            elif 'sat_image_path' in header and 'mask_path' in header:
                image_index = header.index('sat_image_path')
                mask_index = header.index('mask_path')
            else:
                raise ValueError("Column names not found in CSV file.")
            split_index = header.index('split')

            for row in csv_reader:
                if row[split_index] != "train":
                    continue
                image_path = self.csv_file[:-12] + row[image_index]
                mask_path = self.csv_file[:-12] + row[mask_index]

                if not mask_path:
                    break  # Stop if mask_path is empty

                image_paths.append(image_path)
                mask_paths.append(mask_path)

        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        i, j, h, w = RandomResizedCrop.get_params(image,scale=(0.08, 1.0),
                                                  ratio=(0.75,
                                                         1.3333333333333333))
        image = functional.crop(image, i, j, h, w)
        mask = functional.crop(mask, i, j, h, w)

        image = self.resize(image)
        mask = self.resize(mask)

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask
    

def example():
    # Usage example
    csv_files = ["Datasets/" + i + "/metadata.csv" for i in['DeepGlobe', 'ethz-cil-road-segmentation-2023', 'massRD', 'Ottawa-Dataset']]
    data_loaders = []

    for csv_file in csv_files:
        dataset = LazyImageDataset(csv_file)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        data_loaders.append(data_loader)

    for data_loader in data_loaders:
        i = 0
        for images, masks in data_loader:
            if i > 1 :
                break
            plt.imshow(images[0].permute(1, 2, 0))
            plt.axis('off')
            plt.show()

            plt.imshow(masks[0].permute(1, 2, 0))
            plt.axis('off')
            plt.show()
            i = i + 1
            # Process the batch of images and masks
            # ...
