import glob
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
import torchvision.transforms as transforms

class NeighborhoodDataset(Dataset):
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

        self.steering_angles = torch.cat(self.steering_angles)
        # self.steering_angles = (self.steering_angles - self.steering_angles.mean()) / self.steering_angles.std()

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

        # img = cv2.imread(image_path)
        image = Image.open(image_path) 
        trans = transforms.Compose([transforms.ToTensor()])
        img_tensor = trans(image)

        #print image shape
        # img = cv2.resize(img, self.img_dim)
        # img_tensor = torch.from_numpy(img)
        # img_tensor = img_tensor.permute(2, 0, 1)

        
        img_tensor = img_tensor[:3, :, :]  # (3, 144, 256)

        return img_tensor, steering_angle



class HallwayDataset(Dataset):

    def __init__(self, data_path: str, transform=None):
        # normalize steering tensors
        self.steering_angles = torch.from_numpy(np.load(data_path + "split_ctrls\ctrls_2.npy"))
        self.steering_angles = (self.steering_angles - self.steering_angles.mean()) / self.steering_angles.std()

        self.images = torch.from_numpy(np.load(data_path + "split_images\images_2.npy"))
        self.images = torch.permute(self.images, (0, 3, 1, 2))
        self.transform = transform
        # print(f"max={torch.max(self.images[0])}, min={torch.min(self.images[0])}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]        
        steering_angle = self.steering_angles[idx]

        if self.transform:
            # image = (self.transform(image.float())).resize(144, 256)
            image = self.transform(image.float())
        return image, steering_angle

