import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalPixelCNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(ConditionalPixelCNN, self).__init__()

        # Embedding layer for conditioning information
        self.embedding = nn.Embedding(num_classes, num_channels)

        # Main PixelCNN layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

    def forward(self, x, labels):
        # Condition the input on the labels
        if labels is not None:
            labels = self.embedding(labels.long())
            labels = labels.unsqueeze(2).unsqueeze(3)
            labels = labels.repeat(1, 1, x.size(2), x.size(3))
            x = torch.cat((x, labels), dim=1)

        # PixelCNN layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        logits = self.conv4(x)

        return logits