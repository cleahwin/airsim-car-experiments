from torch.utils.data import Dataset, DataLoader
import pandas as pd

def read_data(filelocation):
    df = pd.read_csv(
        filelocation, 
        delimiter = "\t",
        header = 0
    )
    return df


poses_data = read_data("C:/Users/Cleah/Documents/AirSim/2023-01-27-18-52-53/airsim_rec.txt")
print(poses_data)