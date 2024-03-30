from typing import List, Tuple

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


def load_sim_data(
    data_path_list: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load AirSim data from a list of data paths.

    Args:
        data_path_list: List of paths to AirSim data.

    Return:
        Image tensor of shape (N, 3, 144, 256)
            and output tensor of shape (N, 1).
    """
    image_file_names = []
    steering_angles = []

    for path in data_path_list:
        image_paths = path + "/images/"
        timestamps_path = path + "/airsim_rec.txt"

        # Read the collected data into a Pandas DataFrame
        poses_data = pd.read_csv(
                        timestamps_path, 
                        delimiter = "\t",
                        header = 0
                    )

        # Convert the column of image file names to a list
        image_file_names.extend((image_paths + poses_data["ImageFile"]).to_list())

        # Convert the column of steering angle data dataframe to numpy to torch tensor
        steering_angles.append(torch.from_numpy(poses_data[["Steering"]].to_numpy()))

    steering_angles = torch.cat(steering_angles)
    images = torch.empty(0)

    # Creates list of image tensors
    for idx in range(len(steering_angles)):
        image_path = image_file_names[idx]

        image = Image.open(image_path) 
        trans = transforms.Compose([transforms.ToTensor()])
        img_tensor = trans(image)

        img_tensor = img_tensor[:3, :, :]  # (3, 144, 256)


        images.append(img_tensor)

    # Convert list of image tensors to a tensor
    images_tensor = torch.cat(images)

    return (images_tensor, steering_angles)


def load_real_data(
    data_path_list: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load real data from a list of data paths.

    Args:
        data_path_list: List of paths to npy files with real data.

    Return:
        Image tensor of shape (N, 3, 144, 256)
            and output tensor of shape (N, 1).
    """

    steering_angles_list = []
    images_list = []

    for path in data_path_list:
        steering_angles = torch.from_numpy(np.load(path + "split_ctrls\ctrls_2.npy"))
        steering_angles = (steering_angles - steering_angles.mean()) / steering_angles.std()
        steering_angles_list.append(steering_angles)

        images = torch.from_numpy(np.load(path + "split_images\images_2.npy"))
        images = torch.permute(images, (0, 3, 1, 2))
        
        # Normalize all images in images from this data file
        transform = (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        for idx in range(len(images)):
            image = images[idx]
            images[idx] = transform(images.float)

        images_list.append(images)


    return torch.concat(images_list), torch.concat(steering_angles_list)


def shuffle_real_sim_data(
    real_data: Tuple[torch.Tensor, torch.Tensor],
    sim_data: Tuple[torch.Tensor, torch.Tensor],
    sim_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combines real and sim data according to the provided ratio.

    Args:
        real_data: Tuple of input image tensor and output steering angle tensor for real data.
        sim_data: Tuple of input image tensor and output steering angle tensor for sim data.
        sim_ratio: Percentage of how much of sim data to use. Must be between 0 and 1. Model
            will be trained on sim_ratio of sim data and (1-sim_ratio) of real data.
    Return:
        Final combined dataset.
    """
    # Compute length of final data and how much to sample.
    final_data_len = min(len(real_data[0]), len(sim_data[0]))
    sim_data_len = sim_ratio * final_data_len
    real_data_len = final_data_len - sim_data_len

    # Sample real and sim data.
    sample_real_images = real_data[0][:real_data_len]
    sample_real_steering_angle = real_data[1][:real_data_len]

    sample_sim_images = sim_data[0][:sim_data_len]
    sample_sim_steering_angle = sim_data[1][:sim_data_len]

    # Combine real and sim data.

    combined_images = torch.stack((sample_real_images, sample_sim_images), dim=1)
    combined_steering_angles = torch.stack((sample_real_steering_angle, sample_sim_steering_angle), dim=1)

    combined_data = torch.cat(combined_images, combined_steering_angles)

    shuffle_indices = torch.randperm(combined_data.size(0))

    # Shuffle the combined tensor using the random indices
    shuffled_image_sa = combined_data[shuffle_indices]

    # Split the shuffled tensor back into two pairs of tensors
    shuffled_image_pair, shuffled_steering_angle_pair = torch.split(shuffled_image_sa, len(sample_real_images), dim=0)

    return (shuffled_image_pair, shuffled_steering_angle_pair)


class ImageSteeringAngleDataset(Dataset):
    def __init__(self, images: torch.Tensor, steering_angles: torch.Tensor):

        self.images = images
        self.steering_angles = steering_angles

    def __len__(self):
        return len(self.images)    

    def __getitem__(self, idx):
        return self.images[idx], self.steering_angles[idx]
