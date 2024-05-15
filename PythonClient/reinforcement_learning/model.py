import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
        self.tanh = nn.Tanh()

    def forward(self, x):
        # (batch_size, num_channels, height, width)
        # (4, 4, 144, 256)
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.tanh(x)

        return x

# Define ResNet-18 as the backbone
class NeighborhoodResNet(nn.Module):
    def __init__(self, num_classes=1):
        super(NeighborhoodResNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final fully connected layer
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.tanh = nn.Tanh()  # Apply Tanh activation

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.tanh(x)  # Apply Tanh activation
        return x

