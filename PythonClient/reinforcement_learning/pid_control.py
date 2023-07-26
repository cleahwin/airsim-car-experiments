import airsim
import pandas as pd
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


batch_size = 8
# estimated y position for car to be straight
straight_y_pos = 2.26216 * math.pow(10, 5)
# data set of riding normally through neighborhood
data_path = "C:/Users/Cleah/Documents/AirSim/2023-05-09-22-09-34/airsim_rec.txt"

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
print("API Control enabled: %s" % client.isApiControlEnabled())

# Initialize data set
df = pd.read_csv(data_path, delimiter = "\t", header = 0)
steering_angles = []

for i in range(0, 1000):

    for index, row in df.iterrows():
        car_controls = airsim.CarControls()
        car_state = client.getCarState()
        current_y_pos = car_state.kinematics_estimated.position.y_val
        current_x_pos = car_state.kinematics_estimated.position.x_val
        # current_x_quat = car_state.kinematics_estimated.quaterinion.x_val
        r = R.from_quat([row["Q_X"], row["Q_Y"], row["Q_Z"], row["Q_W"]])
        r = r.as_euler('xyz')

        # along track error
        e_at = (
           math.sin(r[2]) * (current_x_pos - row["POS_X"]) + 
            math.cos(r[2]) * (current_y_pos  - row["POS_Y"])
        )
        # cross track error
        e_ct = (
            -1 * math.sin(r[2]) * (current_x_pos - row["POS_X"]) + 
            math.cos(r[2]) * (current_y_pos  - row["POS_Y"])
        )
        # steering angle error
        theta_e = car_controls.steering - r[2]
        k_p = 0.5
        k_d = 1
        print(f"e_at = {e_at}")
        print(f"e_ct = {e_ct}")
        print(f"theta_e = {theta_e}")

        # final calculation
        u = -1 * (k_p * e_ct + k_d * math.sin(r[0]))
        steering_angles.append(u)

        # set car controls
        car_controls.throttle = 0.5
        car_controls.steering = u;
        client.setCarControls(car_controls)

        plt.plot(steering_angles)
        plt.title(f'Plot of Steering Angles Over Time')
        plt.ylabel('Steering Angles')
        plt.xlabel('Time')
        plt.show()
