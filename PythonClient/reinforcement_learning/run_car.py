import airsim
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from model import NeighborhoodRealCNN
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import v2 as transforms_v2


SIMULATOR = False
PATH = r"C:\Users\Cleah\Documents\Projects\University Research\Robot Learning Lab\Simulator\airsim-car-experiments\PythonClient\reinforcement_learning\saved_models"

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print("API Control enabled: %s" % client.isApiControlEnabled())

# Use saved model
cnn = NeighborhoodRealCNN()
cnn.load_state_dict(torch.load(os.path.join(PATH,"1.0-2024-06-27.pth")))
cnn.eval()
data_path = "C:/Users/Cleah/Documents/AirSim/Coastline/2024-04-18-17-22-22"
df = pd.read_csv(data_path + "/airsim_rec.txt", delimiter = "\t", header = 0)

car_controls = airsim.CarControls()
cnn_angles = []
expert_angles = []

# loop through fixed steps and input is from image api
car_controls = airsim.CarControls()
steering_angles = []

# loop through fixed steps and input is from image api
for i in range(0, 10000):
    # get the inputs; data is a list of [inputs, labels]
    # inputs, labels = data
    # inputs, labels = inputs.float(), labels.float()
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]

    # get numpy array
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 

    # reshape array to 3 channel image array H X W X 4
    image = img1d.reshape(1, response.height, response.width, 3)
    image = torch.from_numpy(image)


    image = image.permute(0, 3, 1, 2)
    trans = transforms.Compose([
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        transforms_v2.Resize(size=(140, 252))
    ])
    img_tensor = trans(image)
    img_tensor = img_tensor.float()

    outputs = cnn(img_tensor)
    print(outputs)
    car_controls.throttle = 4.5
    # car_controls.steering = math.radians(outputs.item())
    car_controls.steering = outputs.item()
    steering_angles.append(car_controls.steering)
    client.setCarControls(car_controls)

plt.plot(steering_angles)
plt.plot((pd.read_csv("C:/Users/Cleah/Documents/AirSim/Neighborhood/2023-07-20-12-44-49/airsim_rec.txt", delimiter = "\t", header = 0))['Steering'].to_list())
plt.plot()
plt.title(f'Plot of Steering Angles Over Time')
plt.ylabel('Steering Angles')
plt.xlabel('Time')
plt.show()













# if SIMULATOR:
#     # Use saved model
#     cnn = NeighborhoodCNN()
#     cnn.load_state_dict(torch.load(os.path.join(PATH,"realOnSim.pth")))
#     cnn.eval()
#     print("Eval Model")
    


#     data_path = "C:/Users/Cleah/Documents/AirSim/2023-09-05-10-46-44"
#     df = pd.read_csv(data_path + "/airsim_rec.txt", delimiter = "\t", header = 0)

#     car_controls = airsim.CarControls()
#     cnn_angles = []
#     expert_angles = []
#     # loop through fixed steps and input is from image api
#     for i in range(0, 100):
#         expert_list = pd.read_csv(
#             data_path + "/airsim_rec.txt", delimiter = "\t", header = 0
#             )['Steering'].to_list()
#         dataloader1 = torch.utils.data.DataLoader(NeighborhoodDataset([data_path]), batch_size=1, shuffle=False)
#         for i, data in enumerate(NeighborhoodDataset, 0):
#             inputs, labels = data
#             inputs, labels = inputs.float(), labels.float()

#             outputs = cnn(inputs)

#             car_controls.throttle = 0.5
#             car_controls.steering = math.radians(outputs.item())
#             cnn_angles.append(car_controls.steering)
#             expert_angles.append(expert_list[i])
#             client.setCarControls(car_controls)
# else:
#     # Use saved model
#     cnn = HallwayCNN()
#     cnn.load_state_dict(torch.load(os.path.join(PATH,"realOnSim.pth")))
#     cnn.eval()
#     print("Eval Model")

#     transform = (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
#     data_path = "C:/Users/Cleah/Documents/AirSim/2023-09-05-10-46-44"
#     df = pd.read_csv(data_path + "/airsim_rec.txt", delimiter = "\t", header = 0)

#     car_controls = airsim.CarControls()
#     cnn_angles = []
#     expert_angles = []
#     # loop through fixed steps and input is from image api
#     for i in range(0, 100):
#         expert_list = pd.read_csv(
#             data_path + "/airsim_rec.txt", delimiter = "\t", header = 0
#             )['Steering'].to_list()
#         dataloader1 = torch.utils.data.DataLoader(HallwayDataset("C:\\Users\\Cleah\\Documents\\Projects\\University Research\\Robot Learning Lab\\Simulator\\airsim-car-experiments\\PythonClient\\reinforcement_learning\\balanced_data_split\\", transform=transform), 
#                                                   batch_size=1, 
#                                                   shuffle=False)
#         for i, data in enumerate(dataloader1, 0):
#             inputs, labels = data
#             inputs, labels = inputs.float(), labels.float()

#             outputs = cnn(inputs)

#             car_controls.throttle = 0.5
#             car_controls.steering = math.radians(outputs.item())
#             cnn_angles.append(car_controls.steering)
#             # expert_angles.append(expert_list[i])
#             client.setCarControls(car_controls)


#     # plt.plot(cnn_angles, label="CNN Steering")
#     # # plt.plot(
#     # #     expert_list,
#     # #     label="Expert Steering"
#     # # )
#     # plt.legend(loc="upper right")
#     # plt.plot()
#     # plt.title(f'Plot of Steering Angles Over Time')
#     # plt.ylabel('Steering Angles')
#     # plt.xlabel('Time')
#     # plt.show()
