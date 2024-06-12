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
ROOT_DIR = "/homes/iws/cleahw/AirSim_Research/airsim-car-experiments/PythonClient/"
sim_ratio = args.sim_ratio
sim_environ = args.sim_environ
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.learning_rate
momentum = args.momentum
use_dino = args.use_dino

# Data directories
data_sim_dir = f"{ROOT_DIR}reinforcement_learning/AirSim/{sim_environ}/"
data_real_dir = f"{ROOT_DIR}reinforcement_learning/balanced_data_split_new"

# Define data lists
data_real_list = [f"{data_real_dir}"]
data_sim_list = []

if sim_environ == "Coastline":
    data_sim_list = [f"{data_sim_dir}2024-04-11-15-53-41",
                    f"{data_sim_dir}2024-04-11-16-05-07",
                    f"{data_sim_dir}2024-04-11-16-10-31",
                    f"{data_sim_dir}2024-04-11-16-19-34",
                    f"{data_sim_dir}2024-04-16-11-53-00",
                    f"{data_sim_dir}2024-04-16-15-31-04",
                    f"{data_sim_dir}2024-04-16-22-04-03",
                    f"{data_sim_dir}2024-04-17-08-51-28",
                    f"{data_sim_dir}2024-04-17-08-53-25",
                    f"{data_sim_dir}2024-04-18-17-22-22",
                    f"{data_sim_dir}2024-04-19-13-52-02",
                    f"{data_sim_dir}2024-04-19-13-57-37",
                    f"{data_sim_dir}2024-04-20-16-12-17",
                    f"{data_sim_dir}2024-04-20-16-17-13",
                    f"{data_sim_dir}2024-04-20-16-21-48",
                    f"{data_sim_dir}2024-04-20-16-31-21",
                    f"{data_sim_dir}2024-04-20-16-38-36",
                    f"{data_sim_dir}2024-04-20-16-47-03",
                    f"{data_sim_dir}2024-04-20-16-57-50",
                    f"{data_sim_dir}2024-04-20-17-10-45",
                    f"{data_sim_dir}2024-04-22-14-56-57",
                    f"{data_sim_dir}2024-04-22-14-57-45",
                    f"{data_sim_dir}2024-04-22-15-16-09",
                    f"{data_sim_dir}2024-04-22-15-17-43",
                    f"{data_sim_dir}2024-04-22-15-30-21",
                    f"{data_sim_dir}2024-04-22-15-43-40",
                    f"{data_sim_dir}2024-04-22-15-48-06",
                    f"{data_sim_dir}2024-04-23-14-59-50",
                    f"{data_sim_dir}2024-04-23-16-10-21",
                    f"{data_sim_dir}2024-04-23-16-12-42",
                    f"{data_sim_dir}2024-04-29-08-14-47",
                    f"{data_sim_dir}2024-04-29-08-12-12",
                    f"{data_sim_dir}2024-04-29-08-28-20",
                    f"{data_sim_dir}2024-04-29-08-33-50",
                    f"{data_sim_dir}2024-04-29-08-38-14",
                    f"{data_sim_dir}2024-04-29-08-44-46",
                    f"{data_sim_dir}2024-04-29-08-53-58",
                    f"{data_sim_dir}2024-04-29-08-56-58",
                    f"{data_sim_dir}2024-05-09-18-02-11",
                    f"{data_sim_dir}2024-05-10-14-46-30",
                    f"{data_sim_dir}2024-05-12-17-47-56",
                    f"{data_sim_dir}2024-05-12-17-53-22",
                    f"{data_sim_dir}2024-05-12-17-54-21",
                    f"{data_sim_dir}2024-05-12-17-55-34",
                    f"{data_sim_dir}2024-05-13-07-01-21",
                    f"{data_sim_dir}2024-05-13-07-12-02",
                    f"{data_sim_dir}2024-05-13-07-00-07",
                    f"{data_sim_dir}2024-05-13-07-12-28",
                    f"{data_sim_dir}2024-05-13-07-17-05",
                ]

else:
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
loss_fn = nn.MSELoss()
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)

# Training loop
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
    print(f"Train Loss {epoch_loss}")
    writer.add_scalar('Loss/train', epoch_loss, epoch)

print('Finished Training')

# Save the model
current_date_time = datetime.now()
curr_time = current_date_time.strftime("%Y-%m-%d")
model_dir = f"{ROOT_DIR}saved_models/{sim_ratio}-{curr_time}.pth"
torch.save(cnn.state_dict(), model_dir)

# Evaluation
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
    writer.add_scalar('Loss/test_sim', sim_test_loss, 0)
    print(f"Final Test Loss on SIM {sim_test_loss}")

    for i, data in enumerate(real_testloader, 0):
        inputs, labels = data
        inputs, labels = inputs.float().to(device), labels.float().to(device)  # Move data to device
        outputs = cnn(inputs)
        loss = loss_fn(outputs, labels)
        real_test_loss += loss.item()

    real_test_loss /= len(real_testloader)
    writer.add_scalar('Loss/test_real', real_test_loss, 0)
    print(f"Final Test Loss on REAL {real_test_loss}")

# Close the TensorBoard writer
writer.close()

print("Training complete. Model saved and results logged to TensorBoard.")
