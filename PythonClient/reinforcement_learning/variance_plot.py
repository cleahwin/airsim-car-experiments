

sim_averages = [np.mean(mses) for mses in sim_val_mses_for_sim_ratios_list]
sim_std_devs = [np.std(mses) for mses in sim_val_mses_for_sim_ratios_list]
real_averages = [np.mean(mses) for mses in real_val_mses_for_sim_ratios_list]
real_std_devs = [np.std(mses) for mses in real_val_mses_for_sim_ratios_list]

plt.clf()
# plt.plot(sim_ratios, sim_val_mses_for_sim_ratios, label='Sim Validation')
# plt.plot(sim_ratios, real_val_mses_for_sim_ratios, label='Real Validation')
plt.errorbar(sim_ratios, sim_averages, yerr=sim_std_devs, fmt='-o', label='Sim Validation')
plt.errorbar(sim_ratios, real_averages, yerr=real_std_devs, fmt='-o', label='Real Validation')
plt.title(f'MSEs for Different Ratios Tested on Sim and Real')
plt.ylabel('MSE')
plt.xlabel('Percent of Sim Data')
plt.legend(loc="upper right")
plt.savefig(f"mse_diff_ratios_w_error_bars_epochs={epochs}.png")