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
epochs = 60
learning_rate = 0.001
momentum = 0.9

model_path = "train2.pth"

PATH = r"C:\Users\Cleah\Documents\Projects\University Research\Robot Learning Lab\Simulator\airsim-car-experiments\PythonClient\saved_models"
writer = SummaryWriter()
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
# Initialize data set and data loader and model CNN
dataset = NeighborhoodDataset(data_list)
length = dataset.__len__()
train_length = int(0.8 * length)
test_length = int(length - train_length)
split = random_split(dataset, [train_length, test_length], generator=torch.Generator().manual_seed(42))
trainloader = torch.utils.data.DataLoader(split[0], batch_size=batch_size, shuffle=True)

cnn = Net()

# Optimizer
loss = nn.MSELoss()
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)

running_losses_list = []
for epoch in range(epochs):  # loop over the dataset multiple times
    print(epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.float(), labels.float()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = cnn(inputs)
        # print(f"Input {type(inputs)}")
        # print(f"Output {type(outputs)}")
        # print(f"Labels {type(labels)}")
        # outputs = loss(inputs, labels)
        loss_out = loss(outputs, labels)
        # print(f"LossType {loss.dtype}")
        # print(f"Loss {loss}")
        loss_out.backward()
        optimizer.step()
        running_loss += loss_out.item()
        # print statistics
    
    if i % 2000 == 1999:    # print every 2000 mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    print(f"Running Loss {running_loss / i}")
    running_losses_list.append(running_loss)
    # TODO: Print running_loss after the epoch



print('Finished Training')
torch.save(cnn.state_dict(), os.path.join(PATH, model_path))


# Plot train losses
# TODO: Use maptlotlib.plot to plot the list of running_losses over number of epochs
plt.plot(running_losses_list)
plt.title(f'Loss Curve for {epochs} Epochs on Training Data')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()