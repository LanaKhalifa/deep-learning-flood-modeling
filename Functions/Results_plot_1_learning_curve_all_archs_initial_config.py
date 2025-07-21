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
architectures = ['Arch_02', 'Arch_03', 'Arch_04', 'Arch_05', 'Arch_07', 'Arch_08', 'Arch_09']
trial_num = 'initial_config'
save_path = os.path.join(base_dir, 'all_archs_initial_config.png')

# Initialize dictionaries to store the losses
train_losses = {}
val_losses = {}
dummy_mean_loss = None

# Load the losses for each architecture
for arch in architectures:
    train_loss_path = os.path.join(base_dir, arch, trial_num, 'losses', 'train', 'G_losses_train.pkl')
    val_loss_path = os.path.join(base_dir, arch, trial_num, 'losses', 'val', 'G_losses_val.pkl')
    dummy_loss_path = os.path.join(base_dir, 'Arch_02', trial_num, 'losses', 'dummy', 'mean_dummy_val_loss.pkl')

    with open(train_loss_path, 'rb') as f:
        train_losses[arch] = pickle.load(f)
    
    with open(val_loss_path, 'rb') as f:
        val_losses[arch] = pickle.load(f)
    
    if dummy_mean_loss is None:
        with open(dummy_loss_path, 'rb') as f:
            dummy_mean_loss = pickle.load(f)


# Create the plot
plt.subplots_adjust(wspace=0.5)  # Increase the wspace value to add more space between subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))  # Increase the width from 20 to 24

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Plot training losses
for arch in architectures:
    ax1.plot(train_losses[arch], label=arch, linewidth=2)
    
ax1.set_xlabel('Epoch', fontsize=25,labelpad=20)
ax1.set_ylabel('L1 Loss', fontsize=25, labelpad=22)  # Increase labelpad to move the label further away
ax1.set_ylim(0, 0.3)
ax1.set_title('Training Losses Across Architectures', fontsize=30, pad=20)
ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=20)

# Plot validation losses
for arch in architectures:
    ax2.plot(val_losses[arch], label=arch, linewidth=2)
ax2.axhline(y=dummy_mean_loss, color='green', linestyle='--', label='Mean Validation Dummy Loss', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=25, labelpad=20)
ax2.set_ylim(0, 0.3)
ax2.set_title('Validation Losses Across Architectures', fontsize=30, pad=20)


ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=20)

# Combine legends into one, without duplicating common entries
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Create a set to track already added labels
seen = set()
unique_labels = []
unique_handles = []

for handle, label in zip(handles + handles2, labels + labels2):
    if label not in seen:
        unique_labels.append(label)
        unique_handles.append(handle)
        seen.add(label)

# Adjusting the position of the legend as close as possible to the figure from the downside
#fig.legend(unique_handles, unique_labels, loc='upper center', fontsize=18, ncol=len(unique_labels), bbox_to_anchor=(0.5, 0))
# Adjusting the position of the legend as close as possible to the figure from the downside
fig.legend(unique_handles, unique_labels, loc='upper center', fontsize=25, ncol=len(unique_labels)/2, bbox_to_anchor=(0.5, -0.05))

# Adjust layout to make room for the legend
plt.subplots_adjust(bottom=0.09)


# Save the plot
# plt.suptitle(f'All Architectures - Initial Configurations', fontsize=24)
plt.savefig(save_path, bbox_inches='tight')
plt.show()
plt.close()
