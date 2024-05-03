import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Add back the relu (search google and see the plot) for the calls to conv1 and conv2
# TODO: (LATER) Experiment with the network architecture


# class NeighborhoodRealCNN(nn.Module): 
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 2, 3, 1)  # Reduce channels further
#         self.fc1 = nn.Linear(9216, 4)  # Reduce neurons in the linear layer further

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         return x

class NeighborhoodRealCNN(nn.Module): 

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 8, 5)
        self.fc1 = nn.Linear(7938, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, num_channels, height, width)
        # (4, 4, 144, 256)
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)

        return x
