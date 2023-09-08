import airsim
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from image_dataset import NeighborhoodDataset 
from model import Net
from torchvision import transforms
import torchvision.transforms.functional as F



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

dataset = NeighborhoodDataset(data_list)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
print("Loaded data")

# Use saved model
cnn = Net()
cnn.load_state_dict(torch.load(os.path.join(PATH,"train2.pth")))
cnn.eval()
print("Eval Model")

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print("API Control enabled: %s" % client.isApiControlEnabled())
car_controls = airsim.CarControls()
steering_angles = []
# loop through fixed steps and input is from image api
for i in range(0, 100):
    # get the inputs; data is a list of [inputs, labels]
    # inputs, labels = data
    # inputs, labels = inputs.float(), labels.float()
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    print(f"Responses ==> {responses}")
    response = responses[0]

    # get numpy array
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 

    # reshape array to 3 channel image array H X W X 4
    image = img1d.reshape(1, response.height, response.width, 3)
    image = torch.from_numpy(image)
    print(f"Shape! {image.shape}")


    image = image.permute(0, 3, 1, 2)
    image = image.float()
    # # trans = transforms.Compose([transforms.ToTensor()])
    # image_tensor = F.to_pil_image(image)

    print(type(image))
    outputs = cnn(image)
    car_controls.throttle = 0.5
    car_controls.steering = math.radians(outputs.item())
    steering_angles.append(car_controls.steering)
    client.setCarControls(car_controls)

plt.plot(steering_angles)
plt.plot((pd.read_csv("C:/Users/Cleah/Documents/AirSim/2023-07-20-12-44-49/airsim_rec.txt", delimiter = "\t", header = 0))['Steering'].to_list())
plt.plot()
plt.title(f'Plot of Steering Angles Over Time')
plt.ylabel('Steering Angles')
plt.xlabel('Time')
plt.show()
