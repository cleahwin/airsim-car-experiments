from typing import List, Tuple

import glob
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms as transforms
from torchvision.transforms import v2 as transforms_v2


class NeighborhoodDataset(Dataset):
    def __init__(self, data_path_list: list):

        self.image_file_names = []
        self.steering_angles = []

        for path in data_path_list[:10]:
            image_paths = path + "/images/"
            timestamps_path = path + "/airsim_rec.txt"

            # Read the collected data into a Pandas DataFrame
            poses_data = self.read_data(timestamps_path)

            # Convert the column of image file names to a list
            self.image_file_names.extend((image_paths + poses_data["ImageFile"]).to_list())

            # Convert the column of steering angle data dataframe to numpy to torch tensor
            self.steering_angles.append(torch.from_numpy(poses_data[["Steering"]].to_numpy()))

        self.steering_angles = torch.cat(self.steering_angles)
        

        self.steering_angles = (self.steering_angles - self.steering_angles.mean()) / self.steering_angles.std()

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
        # trans = transforms.Compose([transforms.ToTensor()])
        # img_tensor = trans(image)

        #print image shape
        # img = cv2.resize(img, self.img_dim)
        # img_tensor = torch.from_numpy(img)
        # img_tensor = img_tensor.permute(2, 0, 1)

        
        trans = transforms.Compose([
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            transforms_v2.Resize(size=(140, 252))
        ])
        
        img_tensor = trans(image)

        # Debug: Check transformed tensor shape
        # print("Transformed tensor shape:", img_tensor.shape)        
        
        # Adjust normalization if needed based on the tensor shape
        if img_tensor.shape[0] == 4:
            # Assuming RGBA image, convert to RGB
            img_tensor = img_tensor[:3]  # Keep only RGB channels

        return img_tensor, steering_angle


################################################
# import glob
# import cv2
# import numpy as np
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader, IterableDataset
# import torchvision.transforms as transforms
# from torchvision.transforms import v2 as transforms_v2

# class NeighborhoodDataset(Dataset):
#     def __init__(self, data_path_list: list):

#         self.image_file_names = []
#         self.steering_angles = []

#         for path in data_path_list:
#             image_paths = path + "/images/"
#             timestamps_path = path + "/airsim_rec.txt"

#             # Read the collected data into a Pandas DataFrame
#             poses_data = self.read_data(timestamps_path)

#             # Convert the column of image file names to a list
#             self.image_file_names.extend((image_paths + poses_data["ImageFile"]).to_list())

#             # Convert the column of steering angle data dataframe to numpy to torch tensor
#             self.steering_angles.append(torch.from_numpy(poses_data[["Steering"]].to_numpy()))

#         self.steering_angles = torch.cat(self.steering_angles)
#         # self.steering_angles = (self.steering_angles - self.steering_angles.mean()) / self.steering_angles.std()

#         self.img_dim = (144, 256)    

#     def __len__(self):
#         return len(self.image_file_names)    

#     # Convert the airsim_rec timestamp data to a pandas dataframe given file location
#     def read_data(self, filelocation):
#         df = pd.read_csv(
#             filelocation, 
#             delimiter = "\t",
#             header = 0
#         )
#         return df

#     def __getitem__(self, idx):
#         """
#         TODO:
#         img_path = self.img_paths[idx]
#         # Load img_path
#         steering_angle = 
#         """
#         image_path = self.image_file_names[idx]
#         steering_angle = self.steering_angles[idx]

#         # img = cv2.imread(image_path)
#         image = Image.open(image_path) 
#         trans = transforms.Compose([transforms.ToTensor()])
#         img_tensor = trans(image)

#         #print image shape
#         # img = cv2.resize(img, self.img_dim)
#         # img_tensor = torch.from_numpy(img)
#         # img_tensor = img_tensor.permute(2, 0, 1)

        
#         img_tensor = img_tensor[:3, :, :]  # (3, 144, 256)

#         return img_tensor, steering_angle



# class HallwayDataset(Dataset):

#     def __init__(self, data_path: str, transform=None):
#         # normalize steering tensors
#         self.steering_angles = torch.from_numpy(np.load(data_path + "split_ctrls/ctrls_2.npy"))
#         self.steering_angles = (self.steering_angles - self.steering_angles.mean()) / self.steering_angles.std()

#         self.images = torch.from_numpy(np.load(data_path + "split_images/images_2.npy"))
#         self.images = torch.permute(self.images, (0, 3, 1, 2))
#         self.transform = transform
#         # print(f"max={torch.max(self.images[0])}, min={torch.min(self.images[0])}")

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image = self.images[idx]        
#         steering_angle = self.steering_angles[idx]

#         if self.transform:
#             # image = (self.transform(image.float())).resize(144, 256)
#             image = self.transform(image.float())
#         return image, steering_angle


# def load_sim_data(
#     data_path_list: List[str]
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Load AirSim data from a list of data paths.

#     Args:
#         data_path_list: List of paths to AirSim data.

#     Return:
#         Image tensor of shape (N, 3, 144, 256)
#             and output tensor of shape (N, 1).
#     """
#     image_file_names = []
#     steering_angles = []

#     for path in data_path_list:
#         image_paths = path + "/images/"
#         timestamps_path = path + "/airsim_rec.txt"

#         # Read the collected data into a Pandas DataFrame
#         poses_data = pd.read_csv(
#                         timestamps_path, 
#                         delimiter = "\t",
#                         header = 0
#                     )

#         # Convert the column of image file names to a list
#         image_file_names.extend((image_paths + poses_data["ImageFile"]).to_list())

#         # Convert the column of steering angle data dataframe to numpy to torch tensor
#         steering_angles.append(torch.from_numpy(poses_data[["Steering"]].to_numpy()))


#     steering_angles_tensor = torch.cat(steering_angles)
#     steering_angles_tensor = (steering_angles_tensor - steering_angles_tensor.mean()) / steering_angles_tensor.std()
#     print(f"Steering angle tensor shape: {steering_angles_tensor.shape}")
#     images = []
#     # Creates list of image tensors
#     for idx in range(len(image_file_names)):
#         image_path = image_file_names[idx]

#         image = Image.open(image_path).convert('RGB')
#         trans = transforms.Compose(
#             [
#                 transforms_v2.ToImage(),
#                 transforms_v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
#                 transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#                 transforms.Resize(size=(140, 252))

#             ]
#         )
        
#         img_tensor = trans(image)
#         # print(img_tensor.shape, img_tensor[0].min(), img_tensor[0].max())
#         # img_tensor = img_tensor[:3, :, :]    

#         # transform = (transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
#         # img_tensor = transform(img_tensor)

#         # img_tensor = img_tensor[:3, :, :]  # (3, 144, 256)
#         images.append(img_tensor)
#         # print(torch.min(images[0]), torch.max(images[0]))

#     # Convert list of image tensors to a tensor
#     images_tensor = torch.stack(images, dim=0)
#     # print(f"Images tensor shape data load {images_tensor.shape}")
#     return (images_tensor, steering_angles_tensor)


# def load_real_data(data_path_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Load real data from a list of data paths.

#     Args:
#         data_path_list: List of paths to npy files with real data.

#     Return:
#         Image tensor of shape (N, 3, 144, 256)
#             and output tensor of shape (N, 1).
#     """

#     steering_angles_list = []
#     images_list = []
#     path = data_path_list[0]
#     # NOT LOADING ALL DATA
#     for i in range(1, 15):
#         print(i)
#         if i == 9:
#             continue
#         # Loading steering angles.
#         steering_angles = torch.from_numpy(np.load(path + f"/split_ctrls/ctrls_{i}.npy"))
#         steering_angles = (steering_angles - steering_angles.mean()) / steering_angles.std()
        
#         # Check if all controls are not zero
#         non_zero_mask = torch.any(steering_angles != 0, dim=1)
#         steering_angles = steering_angles[non_zero_mask]
        
#         if len(steering_angles) > 0:  # Check if there are non-zero controls
#             steering_angles_tensor = steering_angles[:, :1]  # Select only the first control
#             images = torch.from_numpy(np.load(path + f"/split_images/images_{i}.npy"))
            
#             images = torch.permute(images, (0, 3, 1, 2))
#             image_transforms = transforms_v2.Compose([
#                 transforms_v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
#                 transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#                 # transforms.Resize(size=(144, 256))
#                 transforms.Resize(size=(140, 252))
#             ])
            
#             # img_tensor = img_tensor[:3]  # Remove the batch dimension
#             # mean, std = img_tensor.mean(), img_tensor.std()  # Calculate mean and std for the entire batch
#             # transform = transforms.Normalize(mean, std)
#             img_tensor = image_transforms(images)

#             # Append the normalized image tensor to the list
#             images_list.append(img_tensor)
#             steering_angles_list.append(steering_angles_tensor)
    
#     # Stack the list of images tensors to form a single tensor
#     images_tensor = torch.cat(images_list, dim=0)
#     print(images_tensor.shape)

#     # images_tensor = images_tensor[non_zero_mask]
#     # steering_angles_tensor = steering_angles_tensor[non_zero_mask]

#     # Concatenate the list of steering angle tensors to form a single tensor
#     steering_angles_tensor = torch.cat(steering_angles_list, dim=0)

#     return (images_tensor, steering_angles_tensor)  


# def shuffle_real_sim_data(
#     real_data: Tuple[torch.Tensor, torch.Tensor],
#     sim_data: Tuple[torch.Tensor, torch.Tensor],
#     sim_ratio: float
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """Combines real and sim data according to the provided ratio.

#     Args:
#         real_data: Tuple of input image tensor and output steering angle tensor for real data.
#         sim_data: Tuple of input image tensor and output steering angle tensor for sim data.
#         sim_ratio: Percentage of how much of sim data to use. Must be between 0 and 1. Model
#             will be trained on sim_ratio of sim data and (1-sim_ratio) of real data.
#     Return:
#         Final combined dataset.
#     """
#     # Compute length of final data and how much to sample.
#     final_data_len = min(len(real_data[1]), len(sim_data[1]))
#     print(len(sim_data[1]), len(real_data[1]), final_data_len)
#     sim_data_len = int(sim_ratio * final_data_len)
#     real_data_len = int(final_data_len - sim_data_len)

#     #TODO: Get random subsamples
#     # Sample real and sim data.
#     sample_real_images = (real_data[0])[:real_data_len]
#     sample_real_steering_angle = (real_data[1])[:real_data_len]

#     sample_sim_images = (sim_data[0])[:sim_data_len]
#     sample_sim_steering_angle = (sim_data[1])[:sim_data_len]

#     print(f"Sim Images Size = {len(sample_sim_images)}, Real Images Size = {len(sample_real_images)}")

#     # Combine real and sim data.
#     combined_images = torch.cat((sample_real_images, sample_sim_images), dim=0)

#     combined_steering_angles = torch.cat((sample_real_steering_angle, sample_sim_steering_angle), dim=0)

#     # combined_data = torch.cat(combined_images, combined_steering_angles)

#     # shuffle_indices = torch.randperm(combined_data.size(0))

#     # # Shuffle the combined tensor using the random indices
#     # shuffled_image_sa = combined_data[shuffle_indices]

#     # # Split the shuffled tensor back into two pairs of tensors
#     # shuffled_image_pair, shuffled_steering_angle_pair = torch.split(shuffled_image_sa, len(sample_real_images), dim=0)
    
#     return (combined_images, combined_steering_angles)


# class ImageSteeringAngleDataset(Dataset):
#     def __init__(self, images: torch.Tensor, steering_angles: torch.Tensor):

#         self.images = images
#         self.steering_angles = steering_angles

#     def __len__(self):
#         return len(self.images)    

#     def __getitem__(self, idx):
#         return self.images[idx], self.steering_angles[idx]
