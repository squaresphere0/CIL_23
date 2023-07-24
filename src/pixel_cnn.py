import torch
from torch import nn
import utils
import math
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def shift_mask(mask):
    return 2 * (mask - 0.5)

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
        '''
        in_ch: number of chanels the block expect at the input
        feat_ch: number of channels generated by the masked convolution
        out_ch: number of channels at the ouput of the block
        kernel_size: size of the masked convolution kernel
        self_referential: if set to true the mask lets the central pixel pass
            information
        '''
        self.conv = nn.Conv2d(in_channels = in_ch,
                              out_channels = feat_ch,
                              kernel_size = kernel_size,
                              padding = math.floor(kernel_size/2))

        self.block = nn.Sequential(nn.ReLU(),
#                                  nn.BatchNorm2d(feat_ch),
                                   nn.Conv2d(in_channels = feat_ch,
                                             out_channels = out_ch,
                                             kernel_size = 1),
                                  nn.ReLU())

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
                                  0).to(device)
    def forward(self, x):
        '''
        The mask is multiplied with the weights of the convolution layer here
        which should set those weights to 0 i.e. hide them.
        '''
        with torch.no_grad():
            self.conv.weight = nn.Parameter(self.conv.weight * self.mask)
        return self.block(self.conv(x))


class PixelCNN(nn.Module):
    '''
    Stacks multiple masked Blocks to achive higher perceptive field and
    complexity. The PixelCNN always should predict an image from itself
    leveraging the fact that for the prediction of each pixel only pixels above
    anf to the right may be used.
    
    After the first layer we can  use self referential masks as only the pixel
    of the input must be hidden from the output and the hidden pixels.
    '''
    def __init__(self, features, in_ch, out_ch, kernels = (5, 3, 3, 3, 3)):
        super().__init__()

        # The tail makes sure the number of channels is correct (1x1
        # convolution)
        self.tail = nn.Conv2d(in_channels = in_ch,
                                         out_channels = features,
                                         kernel_size = 1)

        # This is the layers of masked convolutions.
        self.layer_list = nn.ModuleList()
        self.layer_list.append(Block(features, features, features, 7, False))
        self.layer_list.extend([Block(features, features, features, kern, True) for
                                kern in kernels])

        # Reduce the output to one channel then apply sigmoid to get the
        # probability of a road being present
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
            # Setting a pixel to 0 should be equivalent to not having a signal
            # from said pixel...
            sample = torch.zeros(1,dim,dim)
            for y in range(dim):
                for x in range(dim):
                    '''
                    This loop iterates over all pixels in the image and updates
                    them one at a time from top to bottom (y) and left to right
                    (x). torch.bernoulli is used to sample from the
                    distribution rather than take the probability i.e. average.
                    '''
                    prediction = torch.bernoulli(
                        self.forward(torch.stack([sample],0))[0,:,:,:]
                        )
                    sample[0,y,x] = prediction[0,y,x]
            return sample

    @staticmethod
    def training(epochs, loader, model, optimizer, loss_function):
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


class conditionalPixelCNN(nn.Module):
    '''
    Works similarly to PixelCNN but expects the input to be a concatenation of
    the image to be learned with a "hint" image which would also be present at
    inference time.

    The map is fed through a non referential masked convolution, the hint image
    is fed through a normal convolution with the same kernal size. Outputs are
    concatenated and fed into a masked pipeline like in the PixelCNN.
    '''

    def __init__(self, features, map_ch, cond_ch, kernels = (7, 5, 5, 3, 3),
                 noise = 0.0):
        super().__init__()

        # Used to split the input into autoregressive and hint image
        self.map_ch = map_ch
        self.cond_ch = cond_ch

        # Layers only to be applied to the hint image
        self.conditional_tail = nn.Sequential(
            nn.Conv2d(cond_ch, features, kernels[0],
                      padding=math.floor(kernels[0]/2)),
            nn.ReLU())

        # Layers only to be applied to the autoregressive image
        self.map_tail = nn.Sequential(
            nn.Dropout(noise),
            Block(map_ch, features, features, kernels[0], False))

        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Conv2d(2*features, 2*features, 1))
        self.layer_list.extend([Block(2*features, features, 2*features, kern,
                                      True) for kern in kernels[1:]])

        self.head = nn.Sequential(nn.Conv2d(2*features, map_ch, 1), nn.Sigmoid())


    def forward(self, x):
        # Split the input into map and hint image
        in_map, conditional = torch.split(x, (self.map_ch, self.cond_ch), 1)

        # Apply the respective convolutions and concatenate the results
        # back together
        x = torch.cat((self.map_tail(in_map),
                       self.conditional_tail(conditional)), 1)

        for layer in self.layer_list:
            # For the layers past the first we can use residual layers
            x = layer(x) + x
        return self.head(x)

    def generate_samples(self, num, dim, conditional):
        '''
        Works very similarlly to the code from PixelCNN but predicts for a
        whole batch at once. Expects conditional to contain num images i.e.
        one hint image for every sample to be generated.
        '''
        with torch.no_grad():
            map_sample = torch.zeros(num, 1, dim, dim).to(device)
            for y in range(dim):
                for x in range(dim):
                    index = dim * y + x
                    prediction = torch.bernoulli( self.forward(
                        torch.cat((map_sample, conditional), 1)))
                    prediction = shift_mask(prediction)
                    map_sample[:,:,y,x] = prediction[:,:,y,x]
            return map_sample

    def inference_by_iterative_refinement(self, bias, steps, batchsize, dim, hint):
        '''
        This method uses iterative refinement by feeding a random noise initial
        guess through the model repeatedly instead of predicting pixels one by
        one. This is considerably faster than "proper" prediction pixel by
        pixel.
        '''
        prediction = torch.randn(batchsize, self.map_ch, dim, dim)
#        prediction = torch.zeros(batchsize, self.map_ch, dim, dim)   
        for _ in range(steps):
            prediction = bias(prediction)
            prediction = self(torch.cat((prediction, hint), 1))
            prediction = shift_mask(prediction)
        # we need to shift the output back to the range (0,1)
        return prediction/2 + 0.5



    @staticmethod
    def training(model, loader, optimizer, epochs, name, noise = 0.0):
        model.train()
        losses = []
        for epoch in range(epochs):
            for i, (image, mask) in enumerate(loader):
                image = image.to(device)
                mask = mask.to(device)
                mask = (1-noise) * shift_mask(mask)
                mask += noise * torch.randn(mask.shape).to(device)
                generated = model(torch.cat(
                    (mask, image), 1))

                loss_function = nn.BCELoss()
                loss = loss_function(generated, mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:

                    print("Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, epochs, i + 1, len(loader), loss.item()
                    ))
                    #visualize_images(mask, generated)
                    losses.append(loss.detach())

            torch.save({'model_state_dict': model.state_dict(),
#                        'loss_history': losses
                       }, 'model/'+name+'.pt')

        return losses



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
