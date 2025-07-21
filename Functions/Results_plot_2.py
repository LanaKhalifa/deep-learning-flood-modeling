#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:06:53 2024

@author: lana_k
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from TerrainDownsample_alter import TerrainDownsample_alter
from arch_02 import arch_02
from arch_03 import arch_03
from arch_05 import arch_05
from arch_07 import arch_07
from arch_08 import arch_08
from arch_09 import arch_09

torch.set_default_dtype(torch.float64) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the base directory and architectures
base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/'
architectures = ['Arch_02', 'Arch_03', 'Arch_05', 'Arch_07', 'Arch_08', 'Arch_09']
trial_num = 'initial_config'
val_loader_path = os.path.join(base_dir, 'Dataloaders', 'small_val_loader_Terrain_1_BC_1.pt')
save_path = os.path.join(base_dir, 'all_archs_prediction_comparison.png')

# Load the validation loader
val_loader = torch.load(val_loader_path)

# Get the first batch and the sample at index 22
first_batch = next(iter(val_loader))
terrains, data, labels = first_batch[0], first_batch[1], first_batch[2]
terrain_sample = terrains[22].unsqueeze(0).to(device)
data_sample = data[22].unsqueeze(0).to(device)
label_sample = labels[22].unsqueeze(0).to(device)

# Create a large figure to hold all the subplots
fig = plt.figure(figsize=(45, 70))
gs = GridSpec(7, 4, figure=fig, width_ratios=[1, 4, 4, 4])

# Add column titles
fig.text(0.2, 0.94, 'Predicted (m)', ha='center', fontsize=80)
fig.text(0.5, 0.94, 'Ground Truth (m)', ha='center', fontsize=80)
fig.text(0.8, 0.94, 'Absolute Error', ha='center', fontsize=80)

# Function to create a model instance with default parameters
def create_model(arch):
    shared_terrain_downsample = TerrainDownsample_alter(1, 20, 40, 1).to(device)
    
    if arch == 'Arch_02':
        return arch_02(shared_terrain_downsample).to(device)
    elif arch == 'Arch_03':
        return arch_03(shared_terrain_downsample).to(device)
    elif arch == 'Arch_05':
        return arch_05(shared_terrain_downsample).to(device)
    elif arch == 'Arch_07':
        return arch_07(shared_terrain_downsample).to(device)
    elif arch == 'Arch_08':
        return arch_08(shared_terrain_downsample).to(device)
    elif arch == 'Arch_09':
        return arch_09(shared_terrain_downsample).to(device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

# Loop through each architecture and generate plots
for idx, arch in enumerate(architectures):
    model_path = os.path.join(base_dir, arch, trial_num, 'trained_models', 'model.pth')
    
    # Create the model instance with default parameters
    netG = create_model(arch)
    
    # Load the state dictionary for the model
    netG.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    netG.eval()

    # Generate predictions with the model
    with torch.no_grad():
        prediction_diffs = netG(terrain_sample, data_sample).to(device)
        plot_dir_val = os.path.join(base_dir, arch, trial_num, 'single_plot')
        os.makedirs(plot_dir_val, exist_ok=True)

        # Plot samples and evaluate using the provided function structure
        diff = torch.abs(label_sample - prediction_diffs)
        non_abs_diff = label_sample - prediction_diffs

        # Calculate RAE
        diff_sum = torch.sum(diff, dim=(1, 2, 3))
        dummy_diff_sum = torch.sum(torch.abs(label_sample), dim=(1, 2, 3))
        RAE = diff_sum / dummy_diff_sum

        # Convert tensors to numpy for plotting
        diff = diff.cpu().numpy()
        non_abs_diff = non_abs_diff.cpu().numpy()
        prediction_diffs = prediction_diffs.cpu().numpy()
        label_sample_np = label_sample.cpu().numpy()

        # Add the architecture name on the left
        ax_name = fig.add_subplot(gs[idx+1, 0])
        arch_and_RAE = f'{arch}\n\nRAE = {RAE.item():.2f}'
        ax_name.text(0.5, 0.5, arch_and_RAE, fontsize=80, va='center', ha='center', transform=ax_name.transAxes)
        ax_name.axis('off')

        # Plot on the respective grid positions
        ax_pred = fig.add_subplot(gs[idx+1, 1])
        max_for_colorbar = np.max(np.abs([prediction_diffs.squeeze(), label_sample_np.squeeze()]))
        im_pred = ax_pred.imshow(prediction_diffs.squeeze(), cmap='BrBG', vmin=-max_for_colorbar, vmax=max_for_colorbar)
        ax_pred.axis('off')
        cbar_pred = fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
        cbar_pred.ax.tick_params(labelsize=70)  # Increase tick size

        ax_true = fig.add_subplot(gs[idx+1, 2])
        im_true = ax_true.imshow(label_sample_np.squeeze(), cmap='BrBG', vmin=-max_for_colorbar, vmax=max_for_colorbar)
        ax_true.axis('off')
        cbar_true = fig.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)
        cbar_true.ax.tick_params(labelsize=70)  # Increase tick size

        ax_abs_diff = fig.add_subplot(gs[idx+1, 3])
        m = np.max(diff.squeeze())
        im_abs_diff = ax_abs_diff.imshow(diff.squeeze(), cmap='seismic', vmin=0, vmax=m)
        ax_abs_diff.axis('off')
        cbar_abs_diff = fig.colorbar(im_abs_diff, ax=ax_abs_diff, fraction=0.046, pad=0.04)
        cbar_abs_diff.ax.tick_params(labelsize=70)  # Increase tick size

# Save the combined figure
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust tight_layout to leave space for the titles
plt.savefig(save_path, bbox_inches='tight')
plt.show()
plt.close()
