import cv2
import numpy as np


def resize_npy_images (file_path: str):
    """
    Args:
        file_path -- the path to a .npy file
    """
    np_images = np.load(file_path)
    np_images_resized = np.empty((len(np_images), 144, 256, 3))
    # NOTE: overrides original file path 
    for i in range(len(np_images)):
        # print(f"Shape + {np_images[i].shape} + {np_images[i].transpose(1, 2, 0).shape}")
        # print(np_images[i].shape)
        np_images_resized[i] = cv2.resize(np_images[i], dsize = (256, 144), interpolation=cv2.INTER_AREA)

    np.save(file_path, np_images_resized)

# NOTE: run when you want to resize real data image
resize_npy_images("C:\\Users\\Cleah\\Documents\\Projects\\University Research\\Robot Learning Lab\\Simulator\\airsim-car-experiments\\PythonClient\\reinforcement_learning\\balanced_data_split\\split_images\\images_11.npy")
    