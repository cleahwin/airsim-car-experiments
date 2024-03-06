import torch
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from image_dataset import NeighborhoodDataset, HallwayDataset
from model import NeighborhoodRealCNN, HallwayCNN
from utils_graphs import plot_two_datasets, plot_model_sim_output, plot_loss_curve
import torchvision.transforms as transforms

# true if training using Neighborhood, false if training using Hallway
SIMULATOR = True
# true if a model is continued to be trained
EXISTING_MODEL = True

# Hyperparameters
batch_size = 3
epochs = 40
learning_rate = 0.0001
momentum = 0.9

model_path = "realOnSim.pth"
transform = (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

PATH = r"C:\Users\Cleah\Documents\Projects\University Research\Robot Learning Lab\Simulator\airsim-car-experiments\PythonClient\saved_models"

if (SIMULATOR):
    data_list = [
            "C:/Users/Cleah/Documents/AirSim/2023-07-20-12-44-49",
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
            "C:/Users/Cleah/Documents/AirSim/2023-09-08-11-55-47",
            "C:/Users/Cleah/Documents/AirSim/2023-09-12-10-26-49"
            ]
    # Initialize data set and data loader and model CNN
    dataset = NeighborhoodDataset(data_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

else:
    dataset = HallwayDataset("C:\\Users\\Cleah\\Documents\\Projects\\University Research\\Robot Learning Lab\\Simulator\\airsim-car-experiments\\PythonClient\\reinforcement_learning\\balanced_data_split\\", 
                             transform=transform
                             )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# iter1 = iter(trainloader1)
# all_batches = [next(iter1) for _ in range(len(trainloader1))]
# data1 = torch.cat([batch[1] for batch in all_batches], dim=0)

# iter2 = iter(trainloader2)
# all_batches = [next(iter2) for _ in range(len(trainloader2))]
# data2 = torch.cat([batch[1] for batch in all_batches], dim=0)
# print(data1.shape)
# print(data2.shape)

# plot_two_datasets(data1, torch.reshape(data2, (3260, 1)))
# plot_two_datasets(data1, data2)


dataloader_length = len(dataloader)
cnn = NeighborhoodRealCNN()

# Optimizer
loss = nn.MSELoss()
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)

running_losses_list = []
for epoch in range(epochs):  # loop over the dataset epoch times
    running_loss = 0.0
    print(f"Epoch {epoch}")
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        image, steering_angle = data
        image, steering_angle = image.float(), steering_angle.float()
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = cnn(image)
        loss_out = loss(outputs, steering_angle)
        loss_out.backward()
        optimizer.step()
        running_loss += loss_out.item()
    
    # if i % 2000 == 1999:    # print every 2000 mini-batches
    #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    print(f"Running Loss {running_loss / dataloader_length}")
    running_losses_list.append(float(running_loss / dataloader_length))

print('Finished Training')
torch.save(cnn.state_dict(), os.path.join(PATH, model_path))
print(running_losses_list)

# if (not SIMULATOR):
#     plot_model_sim_output()

# Plot train losses
plot_loss_curve(running_losses_list, epochs)