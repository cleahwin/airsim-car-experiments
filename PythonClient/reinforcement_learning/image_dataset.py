import glob
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class NeighborhoodDataset(Dataset):
    """
    TODO:
    __init__(data_path: str):
        imgs_path = data_path + "/images/"
        labels_file_path = data_path + "/airsim_..txt"
        # Load the labels_file_path
        poses_data = self.read_data()  # poses is a pd.DataFrame
        # Get x (inputs - images)
        poses_data["ImageFile"] --> list
        self.img_paths = ..
        # Get y (outputs - float)
        poses_data["steering_angle"] --> numpy --> torch?

    """
    def __init__(self, data_path_list: list):

        self.image_file_names = []
        self.steering_angles = []

        for path in data_path_list:
            image_paths = path + "/images/"
            timestamps_path = path + "/airsim_rec.txt"

            # Read the collected data into a Pandas DataFrame
            poses_data = self.read_data(timestamps_path)

            # Convert the column of image file names to a list
            self.image_file_names.extend((image_paths + poses_data["ImageFile"]).to_list())

            # Convert the column of steering angle data dataframe to numpy to torch tensor
            self.steering_angles.append(torch.from_numpy(poses_data[["Steering"]].to_numpy()))

        # TODO: concatenate all the torch tensors in self.steering_angles
        self.steering_angles = torch.cat(self.steering_angles)

        # print(f"Image File Names Shape {len(self.image_file_name)}")
        # print(f"Steering Angle Shape {self.steering_angles.shape}")

        
        # self.imgs_path = "C:/Users/Cleah/Documents/AirSim/2023-01-27-18-52-53/images"
        # file_list = glob.glob(self.imgs_path + "*")
        # print(file_list)
        # self.data = []
        # for class_path in file_list:
        #     class_name = class_path.split("/")[-1]
        #     for img_path in glob.glob(class_path + "/*.jpeg"):
        #         self.data.append([img_path, class_name])
        # print(self.data)
        self.img_dim = (144, 256)    

    def __len__(self):
        return len(self.image_file_names)    

    # Convert the airsim_rec timestamp data to a pandas dataframe given file location
    def read_data(self, filelocation):
        df = pd.read_csv(
            filelocation, 
            delimiter = "\t",
            header = 0
        )
        return df

    def __getitem__(self, idx):
        """
        TODO:
        img_path = self.img_paths[idx]
        # Load img_path
        steering_angle = 
        """
        image_path = self.image_file_names[idx]
        steering_angle = self.steering_angles[idx]

        # TODO: check if we need to resize image
        # img = cv2.imread(image_path)
        image = Image.open(image_path) 
        trans = transforms.Compose([transforms.ToTensor()])
        img_tensor = trans(image)

        #print image shape
        # img = cv2.resize(img, self.img_dim)
        # img_tensor = torch.from_numpy(img)
        print(img_tensor.size())  # (4, 144, 256)
        print(f"Shape {img_tensor.shape}")
        # img_tensor = img_tensor.permute(2, 0, 1)

        
        img_tensor = img_tensor[:3, :, :]  # (3, 144, 256)

        print(type(img_tensor))
        print(type(steering_angle))
        print(img_tensor.size())
        print(steering_angle.size())

        return img_tensor, steering_angle



if __name__ == "__main__":
    dataset = NeighborhoodDataset("C:/Users/Cleah/Documents/AirSim/2023-03-12-21-58-13")
    dataset.__getitem__(0)
    print("Done")