import torch
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from image_dataset import NeighborhoodDataset
from model import Net
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
batch_size = 4

model_path = "train2.pth"
PATH = r"C:\Users\Cleah\Documents\Projects\University Research\Robot Learning Lab\Simulator\airsim-car-experiments\PythonClient\saved_models"
data_list = ["C:/Users/Cleah/Documents/AirSim/2023-07-20-12-44-49",
             "C:/Users/Cleah/Documents/AirSim/2023-07-20-15-11-35",
             "C:/Users/Cleah/Documents/AirSim/2023-08-31-12-43-09",
             "C:/Users/Cleah/Documents/AirSim/2023-08-31-17-38-56",
             "C:/Users/Cleah/Documents/AirSim/2023-08-31-17-46-35",
             "C:/Users/Cleah/Documents/AirSim/2023-08-31-17-58-47",
             "C:/Users/Cleah/Documents/AirSim/2023-08-31-18-25-48",
             "C:/Users/Cleah/Documents/AirSim/2023-08-31-18-38-10",
             "C:/Users/Cleah/Documents/AirSim/2023-09-05-10-46-44",
             "C:/Users/Cleah/Documents/AirSim/2023-09-05-17-52-22",
             "C:/Users/Cleah/Documents/AirSim/2023-09-05-18-15-04",
             "C:/Users/Cleah/Documents/AirSim/2023-09-07-11-39-09",
             "C:/Users/Cleah/Documents/AirSim/2023-09-08-08-26-58",
             "C:/Users/Cleah/Documents/AirSim/2023-09-08-08-33-30",
             "C:/Users/Cleah/Documents/AirSim/2023-09-08-08-43-51",
             "C:/Users/Cleah/Documents/AirSim/2023-09-08-09-37-12",
             "C:/Users/Cleah/Documents/AirSim/2023-09-08-11-44-53",
             "C:/Users/Cleah/Documents/AirSim/2023-09-08-11-49-02",
             "C:/Users/Cleah/Documents/AirSim/2023-09-08-11-53-42",
             "C:/Users/Cleah/Documents/AirSim/2023-09-08-11-55-47"
            ]
writer = SummaryWriter()

# Initialize data set and data loader and model CNN; split data
dataset = NeighborhoodDataset(data_list)

length = dataset.__len__()
train_length = int(0.8 * length)
test_length = int(length - train_length)
split = random_split(dataset, [train_length, test_length], generator=torch.Generator().manual_seed(42))


testloader = torch.utils.data.DataLoader(split[1], batch_size=batch_size, shuffle=True)

# Use saved model
cnn = Net()
cnn.load_state_dict(torch.load(os.path.join(PATH,model_path)))
cnn.eval()

# Optimizer
loss = nn.MSELoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)


running_loss = 0;
for i, data in enumerate(testloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    inputs, labels = inputs.float(), labels.float()
    outputs = cnn(inputs)
    print(f"Input {type(inputs)}")
    print(f"Output {type(outputs)}")
    print (f"size!!! {outputs.size()}")
    print(f"Labels {type(labels)}")
    # outputs = loss(inputs, labels)
    loss_out = loss(outputs, labels)
    # print(f"LossType {loss.dtype}")
    # print(f"Loss {loss}")

    running_loss += loss_out.item()
    print(f"Loss out {loss_out.item()}")
    # print statistics

print(f"Running Loss {running_loss / i}")

print('Finished Testing')