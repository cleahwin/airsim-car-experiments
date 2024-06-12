# %load_ext autoreload
# %autoreload 2
import gc
import torch
import os
import importlib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import image_dataset
import psutil

importlib.reload(image_dataset)
# from image_dataset import NeighborhoodDataset
from image_dataset import ImageSteeringAngleDataset, shuffle_real_sim_data, load_real_data, load_sim_data
from model import NeighborhoodRealCNN, NeighborhoodResNet
from utils_graphs import plot_two_datasets, plot_model_sim_output, plot_loss_curve
import torchvision.transforms as transforms
from torch.utils.data import random_split
from datetime import datetime
from PIL import Image


torch.autograd.set_detect_anomaly(True)


NUM_TRIALS = 1
sim_ratios = [0, 0.50, 1.0]
train_mses_for_sim_ratios_list = []
sim_val_mses_for_sim_ratios_list = []
real_val_mses_for_sim_ratios_list = []
for sim_ratio in sim_ratios:
    for trial in range(NUM_TRIALS):
        train_mses_for_sim_ratios = []
        sim_val_mses_for_sim_ratios = []
        real_val_mses_for_sim_ratios = []
        #####################
        ## HYPERPARAMETERS ##
        #####################
        ROOT_DIR = "/homes/iws/cleahw/AirSim_Research/airsim-car-experiments/PythonClient/"
        # Specify ratio of real:sim. 1 - sim_ratio = real_ratio
        # sim_ratio = 0
        # Coastline or Neighborhood
        sim_environ = "Coastline"
        data_sim_dir = f"{ROOT_DIR}reinforcement_learning/AirSim/{sim_environ}/"
        data_real_dir = f"{ROOT_DIR}reinforcement_learning/balanced_data_split_new"
        batch_size = 8
        epochs = 10  #00
        learning_rate = 0.001
        momentum = 0.9
        use_dino = False  
        # dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        ##################
        ## DATA LOADING ##
        ##################


        # Defines list data files
        data_real_list = [f"{data_real_dir}"]
        if sim_environ == "Coastline":
            data_sim_list = [f"{data_sim_dir}2024-04-11-15-53-41",
                            f"{data_sim_dir}2024-04-11-16-05-07",
                            f"{data_sim_dir}2024-04-11-16-10-31",
                            f"{data_sim_dir}2024-04-11-16-19-34",
                            f"{data_sim_dir}2024-04-16-11-53-00",
                            f"{data_sim_dir}2024-04-16-15-31-04",
                            f"{data_sim_dir}2024-04-16-22-04-03",
                            f"{data_sim_dir}2024-04-17-08-51-28",
                            f"{data_sim_dir}2024-04-17-08-53-25",
                            f"{data_sim_dir}2024-04-18-17-22-22",
                            f"{data_sim_dir}2024-04-19-13-52-02",
                            f"{data_sim_dir}2024-04-19-13-57-37",
                            f"{data_sim_dir}2024-04-20-16-12-17",
                            f"{data_sim_dir}2024-04-20-16-17-13",
                            f"{data_sim_dir}2024-04-20-16-21-48",
                            f"{data_sim_dir}2024-04-20-16-31-21",
                            f"{data_sim_dir}2024-04-20-16-38-36",
                            f"{data_sim_dir}2024-04-20-16-47-03",
                            f"{data_sim_dir}2024-04-20-16-57-50",
                            f"{data_sim_dir}2024-04-20-17-10-45",
                            f"{data_sim_dir}2024-04-22-14-56-57",
                            f"{data_sim_dir}2024-04-22-14-57-45",
                            f"{data_sim_dir}2024-04-22-15-16-09",
                            f"{data_sim_dir}2024-04-22-15-17-43",
                            f"{data_sim_dir}2024-04-22-15-30-21",
                            f"{data_sim_dir}2024-04-22-15-43-40",
                            f"{data_sim_dir}2024-04-22-15-48-06",
                            f"{data_sim_dir}2024-04-23-14-59-50",
                            f"{data_sim_dir}2024-04-23-16-10-21",
                            f"{data_sim_dir}2024-04-23-16-12-42",
                            f"{data_sim_dir}2024-04-29-08-14-47",
                            f"{data_sim_dir}2024-04-29-08-12-12",
                            f"{data_sim_dir}2024-04-29-08-28-20",
                            f"{data_sim_dir}2024-04-29-08-33-50",
                            f"{data_sim_dir}2024-04-29-08-38-14",
                            f"{data_sim_dir}2024-04-29-08-44-46",
                            f"{data_sim_dir}2024-04-29-08-53-58",
                            f"{data_sim_dir}2024-04-29-08-56-58",
                            f"{data_sim_dir}2024-05-09-18-02-11",
                            f"{data_sim_dir}2024-05-10-14-46-30",
                            f"{data_sim_dir}2024-05-12-17-47-56",
                            f"{data_sim_dir}2024-05-12-17-53-22",
                            f"{data_sim_dir}2024-05-12-17-54-21",
                            f"{data_sim_dir}2024-05-12-17-55-34",
                            f"{data_sim_dir}2024-05-13-07-01-21",
                            f"{data_sim_dir}2024-05-13-07-12-02",
                            f"{data_sim_dir}2024-05-13-07-00-07",
                            f"{data_sim_dir}2024-05-13-07-12-28",
                            f"{data_sim_dir}2024-05-13-07-17-05",
                        ]

        else:
            data_sim_list =[f"{data_sim_dir}2023-07-20-12-44-49",
                            f"{data_sim_dir}2023-07-20-15-11-35",
                            f"{data_sim_dir}2023-08-31-12-43-09",
                            f"{data_sim_dir}2023-08-31-17-38-56",
                            f"{data_sim_dir}2023-08-31-17-46-35",
                            f"{data_sim_dir}2023-08-31-17-58-47",
                            f"{data_sim_dir}2023-08-31-18-25-48",
                            f"{data_sim_dir}2023-08-31-18-38-10",
                            f"{data_sim_dir}2023-09-05-10-46-44",
                            f"{data_sim_dir}2023-09-05-17-52-22",
                            f"{data_sim_dir}2023-09-05-18-15-04",
                            f"{data_sim_dir}2023-09-07-11-39-09",
                            f"{data_sim_dir}2023-09-08-08-26-58",
                            f"{data_sim_dir}2023-09-08-08-33-30",
                            f"{data_sim_dir}2023-09-08-08-43-51",
                            f"{data_sim_dir}2023-09-08-09-37-12",
                            f"{data_sim_dir}2023-09-08-11-44-53",
                            f"{data_sim_dir}2023-09-08-11-49-02",
                            f"{data_sim_dir}2023-09-08-11-53-42",
                            f"{data_sim_dir}2023-09-08-11-55-47",
                            f"{data_sim_dir}2023-09-12-10-26-49"
                        ]


        # Create instances of your datasets
        # dataset = NeighborhoodDataset(data_sim_list)

        real_data = load_real_data(data_real_list)
        sim_data = load_sim_data(data_sim_list)
        datasets = shuffle_real_sim_data(real_data, sim_data, sim_ratio)

        # # Splits datasets into train and test
        dataset = ImageSteeringAngleDataset(datasets["shuffled_train_images"], datasets["shuffled_train_steering"])
        sim_testset = ImageSteeringAngleDataset(datasets["sim_val_images"], datasets["sim_val_steering"])
        real_testset = ImageSteeringAngleDataset(datasets["real_val_images"], datasets["real_val_steering"])
        # length = dataset.__len__()
        # train_length = int(0.8 * length)
        # test_length = int(length - train_length)
        # split = random_split(dataset, [train_length, test_length], generator=torch.Generator().manual_seed(42))

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        sim_testloader = torch.utils.data.DataLoader(sim_testset, batch_size=batch_size, shuffle=False)
        real_testloader = torch.utils.data.DataLoader(real_testset, batch_size=batch_size, shuffle=False)


        print(f"Training for {epochs} epochs,, batch_size={batch_size} {len(trainloader.dataset)} steps per epoch.")
        print("Finished Data Loading")


        ##############
        ## TRAINING ##
        ##############


        # Define model selection based on the flag
        if use_dino:
            # Use DINO model
            cnn = dinov2_vits14
            # Modify the classifier head to adapt to your task
            num_features = dinov2_vits14.embed_dim    # Get the number of input features from the classifier head
            cnn.head = nn.Linear(num_features, 1)  # Assuming output size is 8 for steering angle prediction
        else:
            # Use custom model
            #cnn = NeighborhoodResNet()
            cnn = NeighborhoodRealCNN()

        cnn.train()

        # Optimizer
        loss = nn.MSELoss()
        optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=momentum)
        running_losses_list = []
        for epoch in range(epochs):  # loop over the dataset epoch times
            running_loss = 0.0
            print(f"EPOCH {epoch}")
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                image, steering_angle = data
                image, steering_angle = image.float(), steering_angle.float()
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = cnn(image)
                loss_out = loss(outputs, steering_angle)
                loss_out.backward()
                optimizer.step()
                running_loss += loss_out.detach()
                if i % 100 == 0:
                    print(f"  Step {i}, loss={loss_out.detach()}")
                del loss_out
                del image, steering_angle
            print(f"Train  Loss {running_loss / len(trainloader.dataset)}\n")
            running_losses_list.append(float(running_loss / len(trainloader.dataset)))

        print('Finished Training')

        # Plot train losses
        plot_loss_curve(running_losses_list, epochs)
        current_date_time = datetime.now()
        curr_time = current_date_time.strftime("%Y-%m-%d")
        model_dir = f"{ROOT_DIR}saved_models/{sim_ratio}-{curr_time}.pth"
        torch.save(cnn.state_dict(), model_dir)


        ################
        ## EVALUATION ##
        ################


        # Use saved model
        
        # cnn.load_state_dict(torch.load(model_dir))
        cnn.eval()

        # Optimizer
        loss = nn.MSELoss()
        running_loss = 0

        with torch.no_grad():
            for i, data in enumerate(sim_testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.float(), labels.float()
                outputs = cnn(inputs)
                loss_out = loss(outputs, labels)
                running_loss += loss_out.item()
                # print(f"Loss out {loss_out.item()}")
                del inputs
                del labels
                del outputs
        sim_test_loss = running_loss / len(sim_testloader.dataset)
        print(f"Final Test Loss on SIM {sim_test_loss}")
        print('Finished Testing Sim')

        # Optimizer
        loss = nn.MSELoss()
        running_loss = 0
        with torch.no_grad():
            for i, data in enumerate(real_testloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.float(), labels.float()
                outputs = cnn(inputs)
                loss_out = loss(outputs, labels)
                running_loss += loss_out.item()
                del inputs
                del labels
                del outputs
                # print(f"Loss out {loss_out.item()}")

        real_test_loss = running_loss / len(real_testloader.dataset)
        print(f"Final Test Loss on REAL {running_loss / len(real_testloader.dataset)}")
        print('Finished Testing Real')

        train_mses_for_sim_ratios.append(running_losses_list[-1])
        sim_val_mses_for_sim_ratios.append(sim_test_loss)
        real_val_mses_for_sim_ratios.append(real_test_loss)

        # Clear state
        del cnn
        del trainloader
        del sim_testloader
        del real_testloader
        gc.collect()

    train_mses_for_sim_ratios_list.append(train_mses_for_sim_ratios)
    sim_val_mses_for_sim_ratios_list.append(sim_val_mses_for_sim_ratios)
    real_val_mses_for_sim_ratios_list.append(real_val_mses_for_sim_ratios)


print("Train MSEs: ", train_mses_for_sim_ratios)
print("Sim Val MSEs: ", sim_val_mses_for_sim_ratios)
print("Real Val MSEs: ", real_val_mses_for_sim_ratios)
sim_averages = [np.mean(mses) for mses in sim_val_mses_for_sim_ratios_list]
sim_std_devs = [np.std(mses) for mses in sim_val_mses_for_sim_ratios_list]
real_averages = [np.mean(mses) for mses in real_val_mses_for_sim_ratios_list]
real_std_devs = [np.std(mses) for mses in real_val_mses_for_sim_ratios_list]

plt.clf()
# plt.plot(sim_ratios, sim_val_mses_for_sim_ratios, labcel='Sim Validation')
# plt.plot(sim_ratios, real_val_mses_for_sim_ratios, label='Real Validation')
plt.errorbar(sim_ratios, sim_averages, yerr=sim_std_devs, fmt='-o', label='Sim Validation')
plt.errorbar(sim_ratios, real_averages, yerr=real_std_devs, fmt='-o', label='Real Validation')
plt.title(f'MSEs for Different Ratios Tested on Sim and Real')
plt.ylabel('MSE')
plt.xlabel('Percent of Sim Data')
plt.legend(loc="upper right")
plt.savefig(f"mse_diff_ratios_w_error_bars_epochs={epochs}.png")


