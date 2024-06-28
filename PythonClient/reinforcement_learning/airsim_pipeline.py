import gc
import torch
import os
import importlib
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import image_dataset
import psutil
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

importlib.reload(image_dataset)
# from image_dataset import NeighborhoodDataset
from image_dataset import ImageSteeringAngleDataset, shuffle_real_sim_data, load_real_data, load_sim_data
from model import NeighborhoodRealCNN, NeighborhoodResNet
from utils_graphs import plot_two_datasets, plot_model_sim_output, plot_loss_curve
import torchvision.transforms as transforms
from torch.utils.data import random_split
from PIL import Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Argument parser for command-line options
parser = argparse.ArgumentParser(description="Training script")
parser.add_argument('--sim_ratio', type=float, default=0.5, help='Ratio of simulation data')
parser.add_argument('--sim_environ', type=str, default="Coastline", help='Simulation environment')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--use_dino', action='store_true', help='Use DINO model')

args = parser.parse_args()

# Hyperparameters
ROOT_DIR = "/gscratch/robotics/cleahw/airsim-car-experiments/PythonClient/reinforcement_learning/"
sim_ratio = args.sim_ratio
sim_environ = args.sim_environ
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
momentum = args.momentum
use_dino = args.use_dino

# Data directories
data_sim_dir = f"{ROOT_DIR}AirSim/{sim_environ}/"
data_real_dir = f"{ROOT_DIR}balanced_data_split_new"

# Define data lists
data_real_list = [f"{data_real_dir}"]
data_sim_list = []

if sim_environ == "Coastline":
    data_sim_list = []
    # Iterate over the items in the given directory
    for item in os.listdir(data_sim_dir):
        # Construct the full path of the item
        item_path = os.path.join(data_sim_dir, item)
        data_sim_list.append(item_path)
    print(len(data_sim_list))
    # data_sim_list = [f"{data_sim_dir}2024-04-11-15-53-41",
    #                 f"{data_sim_dir}2024-04-11-16-05-07",
    #                 f"{data_sim_dir}2024-04-11-16-10-31",
    #                 f"{data_sim_dir}2024-04-11-16-19-34",
    #                 f"{data_sim_dir}2024-04-16-11-53-00",
    #                 f"{data_sim_dir}2024-04-16-15-31-04",
    #                 f"{data_sim_dir}2024-04-16-22-04-03",
    #                 f"{data_sim_dir}2024-04-17-08-51-28",
    #                 f"{data_sim_dir}2024-04-17-08-53-25",
    #                 f"{data_sim_dir}2024-04-18-17-22-22",
    #                 f"{data_sim_dir}2024-04-19-13-52-02",
    #                 f"{data_sim_dir}2024-04-19-13-57-37",
    #                 f"{data_sim_dir}2024-04-20-16-12-17",
    #                 f"{data_sim_dir}2024-04-20-16-17-13",
    #                 f"{data_sim_dir}2024-04-20-16-21-48",
    #                 f"{data_sim_dir}2024-04-20-16-31-21",
    #                 f"{data_sim_dir}2024-04-20-16-38-36",
    #                 f"{data_sim_dir}2024-04-20-16-47-03",
    #                 f"{data_sim_dir}2024-04-20-16-57-50",
    #                 f"{data_sim_dir}2024-04-20-17-10-45",
    #                 f"{data_sim_dir}2024-04-22-14-56-57",
    #                 f"{data_sim_dir}2024-04-22-14-57-45",
    #                 f"{data_sim_dir}2024-04-22-15-16-09",
    #                 f"{data_sim_dir}2024-04-22-15-17-43",
    #                 f"{data_sim_dir}2024-04-22-15-30-21",
    #                 f"{data_sim_dir}2024-04-22-15-43-40",
    #                 f"{data_sim_dir}2024-04-22-15-48-06",
    #                 f"{data_sim_dir}2024-04-23-14-59-50",
    #                 f"{data_sim_dir}2024-04-23-16-10-21",
    #                 f"{data_sim_dir}2024-04-23-16-12-42",
    #                 f"{data_sim_dir}2024-04-29-08-14-47",
    #                 f"{data_sim_dir}2024-04-29-08-12-12",
    #                 f"{data_sim_dir}2024-04-29-08-28-20",
    #                 f"{data_sim_dir}2024-04-29-08-33-50",
    #                 f"{data_sim_dir}2024-04-29-08-38-14",
    #                 f"{data_sim_dir}2024-04-29-08-44-46",
    #                 f"{data_sim_dir}2024-04-29-08-53-58",
    #                 f"{data_sim_dir}2024-04-29-08-56-58",
    #                 f"{data_sim_dir}2024-05-09-18-02-11",
    #                 f"{data_sim_dir}2024-05-10-14-46-30",
    #                 f"{data_sim_dir}2024-05-12-17-47-56",
    #                 f"{data_sim_dir}2024-05-12-17-53-22",
    #                 f"{data_sim_dir}2024-05-12-17-54-21",
    #                 f"{data_sim_dir}2024-05-12-17-55-34",
    #                 f"{data_sim_dir}2024-05-13-07-01-21",
    #                 f"{data_sim_dir}2024-05-13-07-12-02",
    #                 f"{data_sim_dir}2024-05-13-07-00-07",
    #                 f"{data_sim_dir}2024-05-13-07-12-28",
    #                 f"{data_sim_dir}2024-05-13-07-17-05",
    #                 f"{data_sim_dir}2024-05-13-07-18-10",
    #                 f"{data_sim_dir}2024-05-13-07-21-01",
    #                 f"{data_sim_dir}2024-05-13-07-21-28",
    #                 f"{data_sim_dir}2024-05-13-07-23-04",
    #                 f"{data_sim_dir}2024-05-15-08-08-01",
    #                 f"{data_sim_dir}2024-05-15-08-09-21",
    #                 f"{data_sim_dir}2024-05-15-08-11-18",
    #                 f"{data_sim_dir}2024-05-15-08-12-15",
    #                 f"{data_sim_dir}2024-05-15-08-13-44",
    #                 f"{data_sim_dir}2024-05-15-08-15-28",
    #                 f"{data_sim_dir}2024-05-15-08-16-40",
    #                 f"{data_sim_dir}2024-05-15-08-18-31",
    #                 f"{data_sim_dir}2024-05-15-08-21-14",
    #                 f"{data_sim_dir}2024-06-17-14-13-49",
    #                 f"{data_sim_dir}2024-06-17-14-14-47",
    #                 f"{data_sim_dir}2024-06-17-14-15-54",
    #                 f"{data_sim_dir}2024-06-17-14-16-18",
    #                 f"{data_sim_dir}2024-06-17-14-17-01",
    #                 f"{data_sim_dir}2024-06-17-14-18-13",
    #                 f"{data_sim_dir}2024-06-17-14-23-58",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",
    #                 f"{data_sim_dir}",

    #             ]

# Create datasets
real_data = load_real_data(data_real_list)
sim_data = load_sim_data(data_sim_list)
datasets = shuffle_real_sim_data(real_data, sim_data, sim_ratio)

dataset = ImageSteeringAngleDataset(datasets["shuffled_train_images"], datasets["shuffled_train_steering"])
sim_testset = ImageSteeringAngleDataset(datasets["sim_val_images"], datasets["sim_val_steering"])
real_testset = ImageSteeringAngleDataset(datasets["real_val_images"], datasets["real_val_steering"])

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
sim_testloader = torch.utils.data.DataLoader(sim_testset, batch_size=batch_size, shuffle=False)
real_testloader = torch.utils.data.DataLoader(real_testset, batch_size=batch_size, shuffle=False)

# Set up TensorBoard
log_dir = "./logs"
writer = SummaryWriter(log_dir=log_dir)

# Model selection
if use_dino:
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    cnn = dinov2_vits14
    num_features = dinov2_vits14.embed_dim
    cnn.head = nn.Linear(num_features, 1)
else:
    cnn = NeighborhoodRealCNN()

cnn.to(device)  # Move the model to the appropriate device
cnn.train()

# Optimizer and loss function
loss_fn = nn.MSELoss(reduction = 'mean')
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)

# Training loop
train_losses = []
sim_test_losses = []
real_test_losses = []

for epoch in range(epochs):
    running_loss = 0.0
    print(f"EPOCH {epoch + 1}/{epochs}")
    for i, data in enumerate(trainloader, 0):
        images, steering_angles = data
        images, steering_angles = images.float().to(device), steering_angles.float().to(device)  # Move data to device

        # Check for NaNs in data
        if torch.isnan(images).any() or torch.isnan(steering_angles).any():
            print("NaN values found in input data. Skipping this batch.")
            continue

        optimizer.zero_grad()
        outputs = cnn(images)
        loss = loss_fn(outputs, steering_angles)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print(f"  Step {i}, loss={loss.item()}")
    
    epoch_loss = running_loss / len(trainloader)
    train_losses.append(epoch_loss)
    print(f"Train Loss {epoch_loss}")
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    
    # Evaluation after each epoch
    cnn.eval()
    sim_test_loss = 0.0
    real_test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(sim_testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.float().to(device)  # Move data to device
            outputs = cnn(inputs)
            loss = loss_fn(outputs, labels)
            sim_test_loss += loss.item()

        sim_test_loss /= len(sim_testloader)
        sim_test_losses.append(sim_test_loss)
        writer.add_scalar('Loss/test_sim', sim_test_loss, epoch)
        print(f"Epoch {epoch + 1} Test Loss on SIM {sim_test_loss}")

        for i, data in enumerate(real_testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.float().to(device)  # Move data to device
            outputs = cnn(inputs)
            loss = loss_fn(outputs, labels)
            real_test_loss += loss.item()

        real_test_loss /= len(real_testloader)
        real_test_losses.append(real_test_loss)
        writer.add_scalar('Loss/test_real', real_test_loss, epoch)
        print(f"Epoch {epoch + 1} Test Loss on REAL {real_test_loss}")

    cnn.train()

print('Finished Training')

# Save the model
current_date_time = datetime.now()
curr_time = current_date_time.strftime("%Y-%m-%d")
model_dir = f"{ROOT_DIR}saved_models/{sim_ratio}-{curr_time}.pth"
torch.save(cnn.state_dict(), model_dir)

# Close the TensorBoard writer
writer.close()

# Plotting the loss curves and saving the plot
plt.figure()
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), sim_test_losses, label='Sim Test Loss')
plt.plot(range(1, epochs+1), real_test_losses, label='Real Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves')
plt.legend()

# Save the plot
plot_path = f"{ROOT_DIR}saved_plots/loss_curves_{curr_time}.png"
plt.savefig(plot_path)

print(f"Plot saved to {plot_path}")

print("Training complete. Model saved and results logged to TensorBoard.")

# Model selection
if use_dino:
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    cnn = dinov2_vits14
    num_features = dinov2_vits14.embed_dim
    cnn.head = nn.Linear(num_features, 1)
else:
    cnn = NeighborhoodRealCNN()
#    cnn = NeighborhoodResNet()

cnn.to(device)  # Move the model to the appropriate device
cnn.train()

# Optimizer and loss function
loss_fn = nn.MSELoss()
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)

# Training loop
train_losses = []
sim_test_losses = []
real_test_losses = []


for epoch in range(epochs):
    running_loss = 0.0
    print(f"EPOCH {epoch + 1}/{epochs}")
    for i, data in enumerate(trainloader, 0):
        images, steering_angles = data
        images, steering_angles = images.float().to(device), steering_angles.float().to(device)  # Move data to device
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = loss_fn(outputs, steering_angles)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print(f"  Step {i}, loss={loss.item()}")

    epoch_loss = running_loss / len(trainloader)
    train_losses.append(epoch_loss)
    print(f"Train Loss {epoch_loss}")
    writer.add_scalar('Loss/train', epoch_loss, epoch)

    # Evaluation after each epoch
    cnn.eval()
    sim_test_loss = 0.0
    real_test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(sim_testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.float().to(device)  # Move data to device
            outputs = cnn(inputs)
            loss = loss_fn(outputs, labels)
            sim_test_loss += loss.item()

        sim_test_loss /= len(sim_testloader)
        sim_test_losses.append(sim_test_loss)
        writer.add_scalar('Loss/test_sim', sim_test_loss, epoch)
        print(f"Epoch {epoch + 1} Test Loss on SIM {sim_test_loss}")

        for i, data in enumerate(real_testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.float().to(device)  # Move data to device
            outputs = cnn(inputs)
            loss = loss_fn(outputs, labels)
            real_test_loss += loss.item()

        real_test_loss /= len(real_testloader)
        real_test_losses.append(real_test_loss)
        writer.add_scalar('Loss/test_real', real_test_loss, epoch)
        print(f"Epoch {epoch + 1} Test Loss on REAL {real_test_loss}")

    cnn.train()

print('Finished Training')

# Save the model
current_date_time = datetime.now()
curr_time = current_date_time.strftime("%Y-%m-%d")
model_dir = f"{ROOT_DIR}saved_models/{sim_ratio}-{curr_time}.pth"
torch.save(cnn.state_dict(), model_dir)

# Close the TensorBoard writer
writer.close()

# Plotting the loss curves and saving the plot
plt.figure()
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), sim_test_losses, label='Sim Test Loss')
plt.plot(range(1, epochs+1), real_test_losses, label='Real Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves')
plt.legend()

# Save the plot
plot_path = f"{ROOT_DIR}saved_plots/loss_curves_{curr_time}.png"
plt.savefig(plot_path)

print(f"Plot saved to {plot_path}")

print("Training complete. Model saved and results logged to TensorBoard.")

