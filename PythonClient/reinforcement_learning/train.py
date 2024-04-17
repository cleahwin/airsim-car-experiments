# %load_ext autoreload
# %autoreload 2

import torch
import os
import importlib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import image_dataset

importlib.reload(image_dataset)

from image_dataset import ImageSteeringAngleDataset, load_real_data, load_sim_data, shuffle_real_sim_data
from model import NeighborhoodRealCNN
from utils_graphs import plot_two_datasets, plot_model_sim_output, plot_loss_curve
import torchvision.transforms as transforms
from torch.utils.data import random_split

ROOT_DIR = "/homes/iws/cleahw/AirSim_Research/airsim-car-experiments/PythonClient/"

# Specify ratio of real:sim. 1 - sim_ratio = real_ratio
sim_ratio = 0
data_sim_dir = f"{ROOT_DIR}reinforcement_learning/AirSim/"
data_real_dir = f"{ROOT_DIR}reinforcement_learning/balanced_data_split_new"
model_dir = f"{ROOT_DIR}saved_models/sim_model.pth"

batch_size = 2
epochs = 30
learning_rate = 0.0001
momentum = 0.9


# Load data. 

data_real_list = [f"{data_real_dir}"]
data_sim_list = []
data_sim_list =[f"{data_sim_dir}2023-07-20-12-44-49",
                f"{data_sim_dir}2023-07-20-15-11-35",
                f"{data_sim_dir}2023-08-31-12-43-09",
                f"{data_sim_dir}2023-08-31-17-38-56",
                f"{data_sim_dir}2023-08-31-17-46-35",
                f"{data_sim_dir}2023-08-31-17-58-47",
                f"{data_sim_dir}2023-08-31-18-25-48",
                f"{data_sim_dir}2023-08-31-18-38-10",
                f"{data_sim_dir}2023-09-05-10-46-44",
                f"{data_sim_dir}2023-09-05-17-52-22",
                f"{data_sim_dir}2023-09-05-18-15-04",
                f"{data_sim_dir}2023-09-07-11-39-09",
                f"{data_sim_dir}2023-09-08-08-26-58",
                f"{data_sim_dir}2023-09-08-08-33-30",
                f"{data_sim_dir}2023-09-08-08-43-51",
                f"{data_sim_dir}2023-09-08-09-37-12",
                f"{data_sim_dir}2023-09-08-11-44-53",
                f"{data_sim_dir}2023-09-08-11-49-02",
                f"{data_sim_dir}2023-09-08-11-53-42",
                f"{data_sim_dir}2023-09-08-11-55-47",
                f"{data_sim_dir}2023-09-12-10-26-49"
            ]

real_data = load_real_data(data_real_list)
sim_data = load_sim_data(data_sim_list)
print(f"Sim images lenght {sim_data[0].shape} and sa {sim_data[1].shape}")
print(f"Real images lenght {real_data[0].shape} and sa {real_data[1].shape}")

print(f"Max of real images = {torch.max(real_data[0])} and min = {torch.min(real_data[0])}")
print(f"Max of real sa = {torch.max(real_data[1])} and min = {torch.min(real_data[1])}")
print(f"Max of sim images = {torch.max(sim_data[0])} and min = {torch.min(sim_data[0])}")
print(f"Max of sim sa = {torch.max(sim_data[1])} and min = {torch.min(sim_data[1])}")

shuffled_real_sim_data = shuffle_real_sim_data(real_data, sim_data, sim_ratio)

dataset = ImageSteeringAngleDataset(shuffled_real_sim_data[0], shuffled_real_sim_data[1])
length = dataset.__len__()
train_length = int(0.8 * length)
test_length = int(length - train_length)
split = random_split(dataset, [train_length, test_length], generator=torch.Generator().manual_seed(42))


trainloader = torch.utils.data.DataLoader(split[0], batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(split[1], batch_size=batch_size, shuffle=True)



# Loads model
cnn = NeighborhoodRealCNN()

# Optimizer
loss = nn.MSELoss()
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)

running_losses_list = []
for epoch in range(epochs):  # loop over the dataset epoch times
    running_loss = 0.0
    print(f"Epoch {epoch}")
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        image, steering_angle = data
        # print(torch.min(image[0]), torch.max(image[0]))
        # assert False
        # print(f"steering_angle {steering_angle}")
        image, steering_angle = image.float(), steering_angle.float()
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = cnn(image)
        loss_out = loss(outputs, steering_angle)
        loss_out.backward()
        optimizer.step()
        running_loss += loss_out.item()
    
    print(f"Running Loss {running_loss / len(trainloader)}")
    running_losses_list.append(float(running_loss / len(trainloader)))

print('Finished Training')

# Plot train losses
plot_loss_curve(running_losses_list, epochs)

torch.save(cnn, model_dir)


# Use saved model
cnn = NeighborhoodRealCNN()
cnn.load_state_dict(torch.load(os.path.join(model_dir)))
cnn.eval()

# Optimizer
loss = nn.MSELoss()

running_loss = 0;
for i, data in enumerate(testloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    inputs, labels = inputs.float(), labels.float()
    outputs = cnn(inputs)
    loss_out = loss(outputs, labels)

    running_loss += loss_out.item()
    print(f"Loss out {loss_out.item()}")

print(f"Running Loss {running_loss / i}")

print('Finished Testing')