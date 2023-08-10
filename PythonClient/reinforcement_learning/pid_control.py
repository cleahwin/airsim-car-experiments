import airsim
import pandas as pd
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# boolean representing whether to display graphs
GRAPH = False
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
yaw_list = []
posx_ref = []
posy_ref = []
posy_curr = []
posx_curr = []

for i in range(0, 100000):

    car_controls = airsim.CarControls()
    car_state = client.getCarState()

    # Current values of x and y position and orientation
    current_y_pos = car_state.kinematics_estimated.position.y_val
    current_x_pos = car_state.kinematics_estimated.position.x_val
    curr_orientation = car_state.kinematics_estimated.orientation
    
    pos_diff = np.array([current_x_pos - row["POS_X"].item(), current_y_pos - row["POS_Y"]])
    row = df.iloc[(df['POS_Y']-current_x_pos).abs().argsort()[:1]]
    print(f"Row! -- {row}")


    # Numpy arrays of current x-y position and desired x-y position
    pos_curr = np.array([current_x_pos, current_y_pos])
    pos_data = np.array([row["POS_X"].item(), row["POS_Y"].item()])

    

    # current_x_quat = car_state.kinematics_estimated.quaterinion.x_val
    r = R.from_quat([row["Q_X"].item(), row["Q_Y"].item(), row["Q_Z"].item(), row["Q_W"].item()])
    roll, pitch, yaw = r.as_euler('xyz')

    r_curr = R.from_quat([curr_orientation.x_val, 
                            curr_orientation.y_val, 
                            curr_orientation.z_val, 
                            curr_orientation.w_val])
    roll_curr, pitch_curr, yaw_curr = r_curr.as_euler('xyz')

    # Position error
    pos_error = pos_curr - pos_data
    rotation_matrix = np.array([[math.cos(yaw), -1 * math.sin(yaw)],
                                [math.sin(yaw), math.cos(yaw)]])
    # first element corresponds to along track error and second to cross track error
    error_ref_frame = rotation_matrix.T.dot(pos_error)

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

    # heading error
    theta_e = yaw_curr - yaw
    k_p = 0.5
    k_d = 1
    print(f"theta_e = {theta_e}")


    # final calculation
    u = -1 * (k_p * error_ref_frame[1][0] + k_d * car_state.speed * math.sin(theta_e))
    steering_angles.append(u)
    yaw_list.append(yaw)
    posx_ref.append(row["POS_X"])
    posy_ref.append(row["POS_Y"])
    posx_curr.append(current_x_pos)
    posy_curr.append(current_y_pos)

    print(f"steering angle before {u}")

    degree = 2 * math.pi
    k = math.floor(u / degree)
    # angle wrap around
    if (u > degree):
        u = u - k * degree
    elif (u < - degree):
        u = u + k * degree


    # set car controls"
    car_controls.throttle = error_ref_frame[0][0] + 0.5
    car_controls.steering = u
    # car_controls.speed = row["Speed"].item()
    client.setCarControls(car_controls)

if (GRAPH):
    # plt.plot(posx_curr)
    # plt.plot(posx_curr)
    # plt.plot(posx_curr)
    # plt.plot(posx_curr)
    # plt.title(f'Plot of Reference Data for Yaw')
    # plt.ylabel('Yaw')
    # plt.xlabel('Time')
    # plt.show()

    plt.plot(yaw_list)
    plt.title(f'Plot of Reference Data for Yaw')
    plt.ylabel('Yaw')
    plt.xlabel('Time')
    plt.show()

    plt.plot(steering_angles)
    plt.title(f'Plot of Steering Angles Over Time')
    plt.ylabel('Steering Angles')
    plt.xlabel('Time')
    plt.show()