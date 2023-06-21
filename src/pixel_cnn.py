import torch
from torch import nn
import utils
import math

class Block(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, self_referential):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=kernel_size,
                              padding=math.floor(kernel_size/2), bias=False)

        self.block = nn.Sequential(nn.ReLU(),
                                   nn.BatchNorm2d(out_ch))

        with torch.no_grad():
            self.mask = torch.cat((torch.ones(1, math.floor(kernel_size/2)),
                                   torch.tensor([[self_referential]]),
                                   torch.zeros(1, math.floor(kernel_size/2))),
                                  1)
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

test = Block(1, 1, 3, False)
print(test.mask)
print(test.conv.weight)
image = torch.tensor([[[[0,0,0,0,0],
                        [0,0,0,0,0],
                        [0,0,1.0,0,0],
                        [0,0,0,0,0],
                        [0,0,0,0,0]]]])
print(image)
print(test(image))
