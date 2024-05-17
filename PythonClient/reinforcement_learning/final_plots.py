import matplotlib.pyplot as plt


plt.plot([0, 0.5, 0.75, 1], [0.2230, 0.220, 0.1593, 0.2203], label='Sim Validation')
plt.plot([0, 0.5, 0.75, 1], [0.3878, 0.390, 0.4321, 0.4013], label='Real Validation')
plt.title(f'MSEs for Different Ratios Tested on Sim and Real')
plt.ylabel('MSE')
plt.xlabel('Percent of Sim Data')
plt.legend(loc="upper right")
plt.savefig("mse_diff_ratios.png")