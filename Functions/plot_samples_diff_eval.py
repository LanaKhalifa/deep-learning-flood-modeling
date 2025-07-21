#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:34:36 2024

@author: lana_k
"""

import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
import os

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as font_manager


# Set the font to Nimbus Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman']

def plot_samples_diff_eval(epoch, which, Architecture_num, trial_num, prediction_diffs, true_diffs, terrains, water_depth, plot_dir, num_samples=10):
    diff = torch.abs(true_diffs - prediction_diffs)
    non_abs_diff = true_diffs - prediction_diffs
    
    # Calculate Relative Absolute Error
    diff_sum = torch.sum(diff, dim=(1, 2, 3))
    dummy_diff = torch.abs(true_diffs)
    dummy_diff_sum = torch.sum(dummy_diff, dim=(1, 2, 3))
    RAE_batch = diff_sum / dummy_diff_sum
    
    RAE_batch_mean = torch.mean(RAE_batch).item()
    RAE_batch_max = torch.max(RAE_batch).item()
    RAE_batch_median = torch.median(RAE_batch).item()
    
    diff = diff.cpu().numpy()
    non_abs_diff = non_abs_diff.cpu().numpy()

    random_indices = random.sample(range(prediction_diffs.shape[0]), num_samples)
    prediction_diffs = prediction_diffs.cpu().numpy()
    true_diffs = true_diffs.cpu().numpy()

    terrains = terrains.cpu().numpy()
    water_depth = water_depth.cpu().numpy()
    
    for i, sample_index in enumerate(random_indices):
        fig = plt.figure(figsize=(65, 10))  # Increase the width for the additional subplot
        gs = GridSpec(1, 5, width_ratios=[1, 10, 10, 10, 10])  # Adjust GridSpec for the extra text subplot
        
        # First column for RAE value
        ax_text = fig.add_subplot(gs[0, 0])
        ax_text.text(0.5, 0.5, f'{RAE_batch[sample_index].item():.2f}', fontsize=100, ha='center', va='center')
        ax_text.axis('off')  # Hide the axis
        
        # First Row
        max_for_colorbar_1 = np.max(np.abs(prediction_diffs[sample_index]))
        max_for_colorbar_2 = np.max(np.abs(true_diffs[sample_index]))    
        max_for_colorbar = np.max([max_for_colorbar_1,max_for_colorbar_2])
        
        # 1
        ax5 = fig.add_subplot(gs[0, 1])
        im5 = ax5.imshow(prediction_diffs[sample_index].squeeze(), cmap='BrBG', vmin=-max_for_colorbar, vmax=max_for_colorbar)
        #ax5.set_title('Predicted Depth Difference (m)', fontsize=40)
        ax5.axis('off')
        cbar5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=50, length=20, width = 5)  # Set colorbar fontsize

        # 2
        ax6 = fig.add_subplot(gs[0, 2])
        im6 = ax6.imshow(true_diffs[sample_index].squeeze(), cmap='BrBG', vmin=-max_for_colorbar, vmax=max_for_colorbar)
        #ax6.set_title('Ground Truth Depth Difference (m)', fontsize=40)
        ax6.axis('off')
        cbar6 = fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        cbar6.ax.tick_params(labelsize=50, length=20, width = 5)  # Set colorbar fontsize

        # 3
        ax7 = fig.add_subplot(gs[0, 3])
        m = np.max(diff[sample_index])
        im7 = ax7.imshow(diff[sample_index].squeeze(), cmap='seismic', vmin=0, vmax=m)
        #ax7.set_title('Abs Errors abs(True - Pred) (m)', fontsize = 40)
        ax7.axis('off')
        cbar7 = fig.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
        cbar7.ax.tick_params(labelsize=50, length=20, width = 5)  # Set colorbar fontsize

        # 4
        ax_hist1 = fig.add_subplot(gs[0, 4])
        diff_flat = non_abs_diff[sample_index].flatten()
        hist_data, bins, _ = ax_hist1.hist(diff_flat, bins=50, color='black', edgecolor='black')
        #ax_hist1.set_title('Histogram of Errores (True-pred) (m)', fontsize = 40)
        ax_hist1.set_xlim(-m, m)
        ax_hist1.yaxis.set_label_position("right")
        ax_hist1.yaxis.tick_right()
        ax_hist1.tick_params(axis='both', which='major', labelsize=50, length=20, width=5)  # Set tick label size to 14


# =============================================================================
#         plt.suptitle(
#             f'{Architecture_num}_{trial_num}     '
#             f'nu_epochs = {epoch}     '
#             f'dataset = {which}',
#             x=0.6, fontsize=40)
# 
# =============================================================================
        # Save each plot in the designated folder
        plt_path = os.path.join(plot_dir, f'{sample_index}.png')
        plt.savefig(plt_path)

        plt.show()
        plt.close()

    return RAE_batch_mean, RAE_batch_max, RAE_batch_median