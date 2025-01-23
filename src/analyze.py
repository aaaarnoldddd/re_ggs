import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

gs_gwg_data = pd.read_csv("/home/wangqy/Documents/python_test/ckpt/GFP/mutant_7/percentile_0.0_0.3/tik-gamma-1/01_21_2025_23_17/temp-0.1-ngibbs-1000-epochs-10/seed_1.csv")
gwg_data = pd.read_csv("/home/wangqy/Documents/python_test/ckpt/GFP/mutant_7/percentile_0.0_0.3/unsmoothed/01_21_2025_22_47/temp-0.1-ngibbs-1000-epochs-10/seed_1.csv")

gs_gwg_data = gs_gwg_data[['epoch', 'mutant_score']].groupby('epoch')
gwg_data = gwg_data[['epoch', 'mutant_score']].groupby('epoch')

epochs = gs_gwg_data.mean().index.to_numpy()

gs_gwg = [
    gs_gwg_data.mean()['mutant_score'].to_numpy(),
    gs_gwg_data.max()['mutant_score'].to_numpy(),
    gs_gwg_data.min()['mutant_score'].to_numpy()
]

gwg = [
    gwg_data.mean()['mutant_score'].to_numpy(),
    gwg_data.max()['mutant_score'].to_numpy(),
    gwg_data.min()['mutant_score'].to_numpy()
]

# print(gs_gwg)
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

axs[0].plot(epochs, gs_gwg[0], label='gs_gwg Mean', marker='o', linestyle='-', color='blue')
axs[0].plot(epochs, gwg[0], label='gwg Mean', marker='s', linestyle='--', color='red')
axs[0].set_title('Mean Mutant Score per Epoch')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Mean Mutant Score')
axs[0].legend()
axs[0].grid(True)

for i, epoch in enumerate(epochs):
    axs[0].text(epoch, gs_gwg[0][i], f"{gs_gwg[0][i]:.2f}", fontsize=9, ha='center', va='bottom', color='blue')
    axs[0].text(epoch, gwg[0][i], f"{gwg[0][i]:.2f}", fontsize=9, ha='center', va='bottom', color='red')


axs[1].plot(epochs, gs_gwg[1], label='gs_gwg Max', marker='o', linestyle='-', color='green')
axs[1].plot(epochs, gwg[1], label='gwg Max', marker='s', linestyle='--', color='orange')
axs[1].set_title('Max Mutant Score per Epoch')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Max Mutant Score')
axs[1].legend()
axs[1].grid(True)


for i, epoch in enumerate(epochs):
    axs[1].text(epoch, gs_gwg[1][i], f"{gs_gwg[1][i]:.2f}", fontsize=9, ha='center', va='bottom', color='green')
    axs[1].text(epoch, gwg[1][i], f"{gwg[1][i]:.2f}", fontsize=9, ha='center', va='bottom', color='orange')

axs[2].plot(epochs, gs_gwg[2], label='gs_gwg Min', marker='o', linestyle='-', color='purple')
axs[2].plot(epochs, gwg[2], label='gwg Min', marker='s', linestyle='--', color='brown')
axs[2].set_title('Min Mutant Score per Epoch')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Min Mutant Score')
axs[2].legend()
axs[2].grid(True)

for i, epoch in enumerate(epochs):
    axs[2].text(epoch, gs_gwg[2][i], f"{gs_gwg[2][i]:.2f}", fontsize=9, ha='center', va='bottom', color='purple')
    axs[2].text(epoch, gwg[2][i], f"{gwg[2][i]:.2f}", fontsize=9, ha='center', va='bottom', color='brown')

plt.tight_layout()
plt.savefig("/home/wangqy/Documents/python_test/ckpt/GFP/mutant_7/percentile_0.0_0.3/plot.png")
plt.show()

