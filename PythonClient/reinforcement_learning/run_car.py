import airsim
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math


from image_dataset import NeighborhoodDataset 
from model import Net
from torchvision import transforms
import torchvision.transforms.functional as F



PATH = r"C:\Users\Cleah\Documents\Projects\University Research\Robot Learning Lab\Simulator\airsim-car-experiments\PythonClient\saved_models"
data_list = ["C:/Users/Cleah/Documents/AirSim/2023-05-06-12-08-38", 
             "C:/Users/Cleah/Documents/AirSim/2023-01-27-18-52-53",
             "C:/Users/Cleah/Documents/AirSim/2023-05-09-22-09-34",
             "C:/Users/Cleah/Documents/AirSim/2023-07-20-12-44-49",
             "C:/Users/Cleah/Documents/AirSim/2023-07-20-15-11-35"
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
for i in range(0, 10000000000):
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

    print(f"Shape! {image.shape}")

    print(type(image))
    print(f"Shape of Inputs {image.shape}")
    print(f"Dtype {image.dtype}")
    outputs = cnn(image)
    print(f"Output! {image}")
    car_controls.throttle = 0.5
    car_controls.steering = math.radians(outputs.item())
    steering_angles.append(car_controls.steering)
    client.setCarControls(car_controls)

    # plt.plot(steering_angles)
    # plt.title(f'Plot of Steering Angles Over Time')
    # plt.ylabel('Steering Angles')
    # plt.xlabel('Time')
    # plt.show()
