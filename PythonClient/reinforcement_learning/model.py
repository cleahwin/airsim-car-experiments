import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Add back the relu (search google and see the plot) for the calls to conv1 and conv2
# TODO: (LATER) Experiment with the network architecture

class NeighborhoodRealCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 6, 16, 5)
        self.fc1 = nn.Linear(144, 1)
        self.relu = nn.ReLU()


    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # (batch_size, num_channels, height, width)
        # (4, 4, 144, 256)
        x = self.conv1(x)
        x = self.relu(x)
        # (4, 8, 28, 51)
        x = self.conv2(x)
        x = self.relu(x)
        # (4, 6, h, w)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # (4, 6 x h x w)
        x = self.fc1(x)

        return x
    

# NOTE: not in use anymore
class HallwayCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 6, 5)
        self.conv2 = nn.Conv2d(8, 6, 16, 5)
        self.fc1 = nn.Linear(2208, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)

        return x