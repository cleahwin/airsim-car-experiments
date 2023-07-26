import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Add back the relu (search google and see the plot) for the calls to conv1 and conv2
# TODO: (LATER) Experiment with the network architecture

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 6, 16, 5)
        self.fc1 = nn.Linear(144, 1)

    def forward(self, x):
        print("in forward")
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # (batch_size, num_channels, height, width)
        # (4, 4, 144, 256)
        x = self.conv1(x)
        # (4, 8, 28, 51)
        print(x.size())
        x = self.conv2(x)
        # (4, 6, h, w)
        print(x.size())
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # (4, 6 x h x w)
        print(x.size())
        x = self.fc1(x)
        print(x.size())

        return x

        # x = self.pool(F.relu(self.conv1(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
