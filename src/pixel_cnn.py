import torch
from torch import nn
import utils
import math
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

class Block(nn.Module):
    """
    This block consists of the following layers:
        - masked convolution
        - ReLu activation
        - batch norm
        - 1x1 convolution to out_ch number of channels
    """
    def __init__(self, in_ch, feat_ch, out_ch, kernel_size, self_referential):
        super().__init__()
        self.conv = nn.Conv2d(in_channels = in_ch,
                              out_channels = feat_ch,
                              kernel_size = kernel_size,
                              padding = math.floor(kernel_size/2))

        self.block = nn.Sequential(nn.ReLU(),
                                   nn.BatchNorm2d(out_ch),
                                   nn.Conv2d(in_channels = feat_ch,
                                             out_channels = out_ch,
                                             kernel_size = 1))

        with torch.no_grad():
            # this is the middle row of the mask
            self.mask = torch.cat((torch.ones(1, math.floor(kernel_size/2)),
                                   torch.tensor([[self_referential]]),
                                   torch.zeros(1, math.floor(kernel_size/2))),
                                  1)
            # all rows above should be 1 all below 0
            self.mask = torch.cat((torch.ones(math.floor(kernel_size/2),
                                              kernel_size),
                                   self.mask,
                                   torch.zeros(math.floor(kernel_size/2),
                                               kernel_size)),
                                  0)
    def forward(self, x):
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)
        return self.block(self.conv(x))


class PixelCNN(nn.Module):

    def __init__(self, features, in_ch, out_ch, kernels = (5, 3, 3, 3, 3)):
        super().__init__()

        self.tail = nn.Conv2d(in_channels = in_ch,
                                         out_channels = features,
                                         kernel_size = 1)

        self.layer_list = nn.ModuleList()
        self.layer_list.append(Block(features, features, features, 7, False))
        self.layer_list.extend([Block(features, features, features, kern, True) for
                                kern in kernels])

        self.head = nn.Sequential(nn.Conv2d(in_channels = features,
                                         out_channels = out_ch,
                                         kernel_size = 1),
                                 nn.Sigmoid())

    def forward(self, x):
        x = self.tail(x)
        for layer in self.layer_list:
            x = layer(x)
        return self.head(x)

    def generate_samples(self, num, dim):
        samples = [self.generate_sample(dim) for _ in range(num)]
        return torch.stack(samples, 0)

    def generate_sample(self, dim):
        with torch.no_grad():
            sample = torch.zeros(1,dim,dim)
            for y in range(dim):
                for x in range(dim):
                    index = dim * y + x
                    prediction = torch.bernoulli(
                        self.forward(torch.stack([sample],0))[0,:,:,:]
                        )
                    sample[0,y,x] = prediction[0,y,x]
            return sample

class conditionalPixelCNN(nn.Module):

    def __init__(self, features, map_ch, cond_ch, kernels = (7, 5, 5, 3, 3)):
        super().__init__()

        self.map_ch = map_ch
        self.cond_ch = cond_ch

        self.conditional_tail = nn.Conv2d(cond_ch, features, kernels[0],
                                          padding='same')
        self.map_tail = Block(map_ch, features, features, kernels[0], False)

        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Conv2d(2*features, features, 1))
        self.layer_list.extend([Block(features, features*2, features, kern,
                                      True) for kern in kernels[1:]])

        self.head = nn.Sepuential(nn.Conv2d(features, map_ch, 1), nn.Sigmoid())


    def forward(self, x):
        in_map, conditional = torch.split(x, (self.map_ch, self.cond_ch), 1)

        x = torch.cat((self.map_tail(in_map),
                       self.conditional_tail(conditional)), 1)

        for layer in self.layer_list:
            x = layer(x)
        return self.head(x)

    def generate_samples(self, num, dim, conditional):
        with torch.no_grad():
            map_sample = torch.zeros(num, 1, dim, dim)

            for y in range(dim):
                for x in range(dim):
                    index = dim * y + x
                    prediction = torch.bernoulli( self.forward(
                        torch.cat((map_sample, conditional), 1)))
                    map_sample[:,:,y,x] = prediction[:,:,y,x]
            return map_sample






def train(epochs, loader, model, optimizer, loss_function):
    model.train()
    outputs = []
    losses = []
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for i, (image, _) in enumerate(loader):
            # Output of Autoencoder
            reconstructed = model(image)

            # Calculating the loss function
            loss = loss_function(reconstructed, image)

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print intermediate outputs for debugging
            if i % 231 == 0:
                print("Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, epochs, i + 1, len(loader), loss.item()
                ))
                # Visualize example images and their reconstructed versions
                # visualize_images(image, reconstructed)

            # Storing the losses in a list for plotting
            losses.append(loss.detach())
        outputs.append((epochs, image, reconstructed))

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    plt.plot(losses)
    plt.show()

def visualize_images(original, reconstructed):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    for i in range(4):
        axes[0, i].imshow(original[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")
        axes[1, i].imshow(reconstructed[i].detach().numpy().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")
    plt.tight_layout()
    plt.show()
