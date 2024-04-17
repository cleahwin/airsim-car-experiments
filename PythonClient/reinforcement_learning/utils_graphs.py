import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt



def plot_two_datasets (data_1=None, data_2=None):
    plt.plot(data_1, color='red', label='Simulator')
    plt.plot(data_2, color='blue', label='Real')
    plt.title(f'Simulator vs. Real Data Steering Angles')
    plt.ylabel('Steering Angle')
    plt.xlabel('Time Point')
    plt.legend(loc="upper right")
    plt.show()

def plot_model_sim_output (model=None, sim=None):
    plt.plot(model, color='red', label='Model Output')
    plt.plot(sim, color='blue', label='Simulation Baseline')
    plt.title(f'Model Output Compared to Baseline')
    plt.ylabel('Steering Angle')
    plt.xlabel('Data Point')
    plt.legend(loc="upper right")
    plt.show()

def plot_loss_curve (running_loss, epochs):
    print(running_loss)
    plt.plot(running_loss)
    plt.title(f'Loss Curve for {epochs} Epochs on Training Data')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(f"loss_curve_{epochs}.png")