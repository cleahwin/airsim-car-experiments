import airsim
import pandas as pd
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


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
        # Numpy arrays of current x-y position and desired x-y position
        pos_curr = np.array([[current_x_pos], 
                              [current_y_pos]])
        pos_data = np.array([[row["POS_X"]], 
                              [row["POS_Y"]]])

        # current_x_quat = car_state.kinematics_estimated.quaterinion.x_val
        r = R.from_quat([row["Q_X"], row["Q_Y"], row["Q_Z"], row["Q_W"]])
        roll, pitch, yaw = r.as_euler('xyz')

        # Position error
        pos_error = pos_curr - pos_data
        rotation_matrix = np.array([[math.cos(car_controls.steering), -1 * math.sin(car_controls.steering)],
                                  [math.sin(car_controls.steering), math.cos(car_controls.steering)]])
        # first element corresponds to along track error and second to cross track error
        track_error_matrix = np.matmul(rotation_matrix, pos_error)

        # # along track error
        # e_at = (
        #    math.sin(yaw) * (current_x_pos - row["POS_X"]) + 
        #     math.cos(yaw) * (current_y_pos  - row["POS_Y"])
        # )
        # # cross track error
        # e_ct = (
        #     -1 * math.sin(yaw) * (current_x_pos - row["POS_X"]) + 
        #     math.cos(yaw) * (current_y_pos  - row["POS_Y"])
        # )
        # steering angle error
        theta_e = car_controls.steering - yaw
        k_p = 0.5
        k_d = 1
        print(f"e_at = {track_error_matrix[0][0]}")
        print(f"e_ct = {track_error_matrix[1][0]}")
        print(f"theta_e = {theta_e}")

        # final calculation
        u = -1 * (k_p * track_error_matrix[1][0] + k_d * car_state.speed * math.sin(yaw))
        steering_angles.append(u)
        print(f"steering angle {u}")
        # set car controls
        car_controls.throttle = 0.5
        car_controls.steering = u;
        client.setCarControls(car_controls)

        plt.plot(steering_angles)
        plt.title(f'Plot of Steering Angles Over Time')
        plt.ylabel('Steering Angles')
        plt.xlabel('Time')
        plt.show()
