#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 17:01:50 2024

@author: lana_k
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:29:47 2024

@author: lana_k
"""

import os
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as font_manager

# Set the font to Nimbus Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman']

# Define the base directory and architectures
base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/'
architectures = {
    'Arch_02': 'No Downsampling Convolutions',
    'Arch_03': 'Simplified UNet',
    'Arch_04': 'No Downsampling Convolutions with Self-Attention',
    'Arch_05': 'Classic UNet',
    'Arch_07': 'Encoder-Decoder with Self-Attention',
    'Arch_08': 'UNet and ResNet Modified',
    'Arch_09': 'Encoder-Decoder with Large Convolutions'
}
trial_num = 'initial_config'
save_path = os.path.join(base_dir, 'all_archs_initial_config.png')

# Initialize dictionaries to store the losses
train_losses = {}
val_losses = {}
dummy_mean_loss = None

# Load the losses for each architecture
for arch_key, arch_name in architectures.items():
    train_loss_path = os.path.join(base_dir, arch_key, trial_num, 'losses', 'train', 'G_losses_train.pkl')
    val_loss_path = os.path.join(base_dir, arch_key, trial_num, 'losses', 'val', 'G_losses_val.pkl')
    dummy_loss_path = os.path.join(base_dir, 'Arch_02', trial_num, 'losses', 'dummy', 'mean_dummy_val_loss.pkl')

    with open(train_loss_path, 'rb') as f:
        train_losses[arch_name] = pickle.load(f)
    
    with open(val_loss_path, 'rb') as f:
        val_losses[arch_name] = pickle.load(f)
    
    if dummy_mean_loss is None:
        with open(dummy_loss_path, 'rb') as f:
            dummy_mean_loss = pickle.load(f)

# Create the plot
plt.subplots_adjust(wspace=0.5)  # Increase the wspace value to add more space between subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))  # Increase the width from 20 to 24

# Plot training losses
for arch_name in architectures.values():
    ax1.plot(train_losses[arch_name], label=arch_name, linewidth=2)
    
ax1.set_xlabel('Epoch', fontsize=35, labelpad=20)
ax1.set_ylabel('L1 Loss', fontsize=35, labelpad=22)  # Increase labelpad to move the label further away
ax1.set_ylim(0, 0.3)
ax1.set_title('Training Set', fontsize=40, pad=20)
ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=30)

# Plot validation losses
for arch_name in architectures.values():
    ax2.plot(val_losses[arch_name], label=arch_name, linewidth=2)
dummy_handle = ax2.axhline(y=dummy_mean_loss, color='green', linestyle='--', label='Mean Validation Dummy Loss', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=35, labelpad=20)
ax2.set_ylim(0, 0.3)
ax2.set_title('Validation Set', fontsize=40, pad=20)

ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=30)

# Combine the handles and labels from the first subplot with the dummy mean validation loss label
handles, labels = ax1.get_legend_handles_labels()
handles.append(dummy_handle)
labels.append('Mean Validation Dummy Loss')

# Adjusting the position of the legend as close as possible to the figure from the downside
fig.legend(handles, labels, loc='upper center', fontsize=35, ncol=2, bbox_to_anchor=(0.5, -0.05))

# Adjust layout to make room for the legend
plt.subplots_adjust(bottom=0.09)

# Save the plot
plt.savefig(save_path, bbox_inches='tight')
plt.show()
plt.close()
