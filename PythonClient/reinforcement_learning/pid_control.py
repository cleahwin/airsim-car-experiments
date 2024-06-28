import airsim
import pandas as pd
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import random
import os

# data set of riding normally through neighborhood
data_path = "C:/Users/Cleah/Documents/AirSim/Coastline/2024-04-18-17-22-22/airsim_rec.txt"
df = pd.read_csv(data_path, delimiter="\t", header=0)
df_first_100 = df.head(25)

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)

print("API Control enabled: %s" % client.isApiControlEnabled())

n_runs = 1000

for run_num in range(n_runs):
    print(f"Starting run {run_num + 1}/{n_runs}")

    # Randomly select a starting position from the data
    random_row = df_first_100.sample().iloc[0]
    random_pos = airsim.Vector3r(random_row["POS_X"], random_row["POS_Y"], random_row["POS_Z"])
    #initial_pose = airsim.Pose(random_pos, random_orientation)

    # Start AirSim recording
    recording_folder = os.path.join(r"C://Users//Cleah//Documents//AirSim", f"run_{run_num + 1}")
    print(recording_folder)
    client.startRecording()
    start_pose = client.simGetVehiclePose()

    random_angle_deg = random.uniform(-0.1, 0.01)
    random_angle_rad = math.radians(random_angle_deg)
    print(random_angle_deg)

    # Set initial pose for the car
    # initial_pose = airsim.Pose(airsim.Vector3r(0, 0 , -1), airsim.to_quaternion(0, 0, random_angle_rad))
    initial_pose = airsim.Pose(random_pos, airsim.to_quaternion(0, 0, random_angle_rad))
    client.simSetVehiclePose(pose=initial_pose, ignore_collision=True)

    # Debugging output
    print(f"Requested Initial Pose: {initial_pose}")

    # Get current vehicle pose after setting it
    current_pose = client.simGetVehiclePose()
    print(f"Current Vehicle Pose: {current_pose}")



    # Initialize data set
    df = pd.read_csv(data_path, delimiter = "\t", header = 0)
    steering_angles = []
    yaw_list = []
    posx_ref = []
    posy_ref = []
    posy_curr = []
    posx_curr = []

    for i in range(0, 1500):
        collision_info = client.simGetCollisionInfo()
            
        if collision_info.has_collided:
            print("Collision detected!")
            client.stopRecording()
            print("Setting beginning pose!")
            client.simSetVehiclePose(pose=start_pose, ignore_collision=True)
        print(f"In PID loop {i}")

        car_controls = airsim.CarControls()
        car_state = client.getCarState()


        ###################
        ## Intial Values ##
        ###################

        # Current values of x and y position and orientation
        current_y_pos = car_state.kinematics_estimated.position.y_val
        current_x_pos = car_state.kinematics_estimated.position.x_val
        curr_orientation = car_state.kinematics_estimated.orientation
        curr_speed = car_state.speed
        
        error_dist = []
        # compute distance fromm current x,y and all x,y
        pos_curr = np.array([current_x_pos, current_y_pos])
        
        ######################
        ## Find Closest Row ##
        ######################

        # computes array of error distances
        for index, row in df.iterrows():
            pos_diff = np.array([row["POS_X"], row["POS_Y"]]) - pos_curr
            error_dist.append(np.linalg.norm(pos_diff))
        
        index = 0
        smallest_value = error_dist[0]
        for idx, value in enumerate(error_dist):
            if value < smallest_value:
                index = idx
                smallest_value = value

        row = df.iloc[index]

        # row = df.iloc[(df['POS_Y']-current_y_pos).abs().argsort()[:1]]


        ##################
        ## Steering PID ##
        ##################

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

        # heading error
        theta_e = yaw_curr - yaw
        k_p = 0.5
        k_d = 1


        # final calculation
        u = -1 * (k_p * error_ref_frame[1] + k_d * car_state.speed * math.sin(theta_e))
        steering_angles.append(u)
        yaw_list.append(yaw)

        # Deals with carry overs for angle values
        degree = 2 * math.pi
        k = math.floor(u / degree)
        if (u > degree):
            u = u - k * degree
        elif (u < - degree):
            u = u + k * degree


        ##################
        ## Throttle PID ##
        ##################

        velocity = np.array([curr_speed])

        p_err = 0
        v_err = 0

        k_pt = 1
        k_v = 1

        throttle = -1 * k_pt * p_err - k_v * v_err


        ######################
        ## Set Car Controls ##
        ######################

        # set car controls
        car_controls.throttle = 0.5
        car_controls.steering = u
        # car_controls.speed = row["Speed"].item()
        client.setCarControls(car_controls)


    client.stopRecording()
    print("Setting beginning pose!")
    client.simSetVehiclePose(pose=start_pose, ignore_collision=True)