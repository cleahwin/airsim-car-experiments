import airsim
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from image_dataset import HallwayDataset, NeighborhoodDataset
from model import HallwayCNN, NeighborhoodCNN
from torchvision import transforms
import torchvision.transforms.functional as F


SIMULATOR = False
PATH = r"C:\Users\Cleah\Documents\Projects\University Research\Robot Learning Lab\Simulator\airsim-car-experiments\PythonClient\saved_models"

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print("API Control enabled: %s" % client.isApiControlEnabled())


if SIMULATOR:
    # Use saved model
    cnn = NeighborhoodCNN()
    cnn.load_state_dict(torch.load(os.path.join(PATH,"realOnSim.pth")))
    cnn.eval()
    print("Eval Model")


    data_path = "C:/Users/Cleah/Documents/AirSim/2023-09-05-10-46-44"
    df = pd.read_csv(data_path + "/airsim_rec.txt", delimiter = "\t", header = 0)

    car_controls = airsim.CarControls()
    cnn_angles = []
    expert_angles = []
    # loop through fixed steps and input is from image api
    for i in range(0, 100):
        expert_list = pd.read_csv(
            data_path + "/airsim_rec.txt", delimiter = "\t", header = 0
            )['Steering'].to_list()
        dataloader1 = torch.utils.data.DataLoader(NeighborhoodDataset([data_path]), batch_size=1, shuffle=False)
        for i, data in enumerate(NeighborhoodDataset, 0):
            inputs, labels = data
            inputs, labels = inputs.float(), labels.float()

            outputs = cnn(inputs)

            car_controls.throttle = 0.5
            car_controls.steering = math.radians(outputs.item())
            cnn_angles.append(car_controls.steering)
            expert_angles.append(expert_list[i])
            client.setCarControls(car_controls)
else:
    # Use saved model
    cnn = HallwayCNN()
    cnn.load_state_dict(torch.load(os.path.join(PATH,"realOnSim.pth")))
    cnn.eval()
    print("Eval Model")

    transform = (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    data_path = "C:/Users/Cleah/Documents/AirSim/2023-09-05-10-46-44"
    df = pd.read_csv(data_path + "/airsim_rec.txt", delimiter = "\t", header = 0)

    car_controls = airsim.CarControls()
    cnn_angles = []
    expert_angles = []
    # loop through fixed steps and input is from image api
    for i in range(0, 100):
        expert_list = pd.read_csv(
            data_path + "/airsim_rec.txt", delimiter = "\t", header = 0
            )['Steering'].to_list()
        dataloader1 = torch.utils.data.DataLoader(HallwayDataset("C:\\Users\\Cleah\\Documents\\Projects\\University Research\\Robot Learning Lab\\Simulator\\airsim-car-experiments\\PythonClient\\reinforcement_learning\\balanced_data_split\\", transform=transform), 
                                                  batch_size=1, 
                                                  shuffle=False)
        for i, data in enumerate(dataloader1, 0):
            inputs, labels = data
            inputs, labels = inputs.float(), labels.float()

            outputs = cnn(inputs)

            car_controls.throttle = 0.5
            car_controls.steering = math.radians(outputs.item())
            cnn_angles.append(car_controls.steering)
            # expert_angles.append(expert_list[i])
            client.setCarControls(car_controls)


    # plt.plot(cnn_angles, label="CNN Steering")
    # # plt.plot(
    # #     expert_list,
    # #     label="Expert Steering"
    # # )
    # plt.legend(loc="upper right")
    # plt.plot()
    # plt.title(f'Plot of Steering Angles Over Time')
    # plt.ylabel('Steering Angles')
    # plt.xlabel('Time')
    # plt.show()
