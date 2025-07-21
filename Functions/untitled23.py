#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:00:35 2024

@author: lana_k
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

# Define directories
RAE_dir = "/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/RAE_test/"
num_cells_dir = "/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/num_cells/"

# Initialize lists for RAE groups
RAE_t0 = []
RAE_t1 = []
RAE_t2 = []
RAE_t3 = []

# Load sizes (number of cells) for each project and plan combination
sizes_t0 = []
sizes_t1 = []
sizes_t2 = []
sizes_t3 = []

# Group RAEs based on time step
for file_path in glob.glob(os.path.join(RAE_dir, "*_RAE.npy")):
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    t_value = int(parts[2])
    
    rae_value = np.load(file_path).item()
    num_cells_file_path = os.path.join(num_cells_dir, f'{parts[0]}_{parts[1]}_num_cells.npy')
    num_cells_value = np.load(num_cells_file_path)
    
    if t_value == 0:
        RAE_t0.append(rae_value)
        sizes_t0.append(num_cells_value)
        
    elif t_value == 70:
        RAE_t1.append(rae_value)
        sizes_t1.append(num_cells_value)

    elif t_value == 140:
        RAE_t2.append(rae_value)
        sizes_t2.append(num_cells_value)

    elif t_value == 210:
        RAE_t3.append(rae_value)
        sizes_t3.append(num_cells_value)

# Define x-axis values for each group to avoid overlap
x_t0 = np.random.normal(1, 0.001, len(RAE_t0))  # Centered around 1 with small scatter
x_t1 = np.random.normal(2, 0.001, len(RAE_t1))  # Centered around 2 with small scatter
x_t2 = np.random.normal(3, 0.001, len(RAE_t2))  # Centered around 3 with small scatter
x_t3 = np.random.normal(4, 0.001, len(RAE_t3))  # Centered around 4 with small scatter

# Create figure
plt.figure(figsize=(30, 15))

# Create a box plot for each group with thicker lines
boxprops = dict(facecolor='none', edgecolor='black', linewidth=2)  # Set linewidth here
whiskerprops = dict(color='black', linewidth=3)  # Set whisker line thickness
capprops = dict(color='black', linewidth=3)  # Set cap line thickness
medianprops = dict(color='red', linewidth=3)  # Set median line thickness

# Create a box plot for each group with matching colors
plt.boxplot([RAE_t0, RAE_t1, RAE_t2, RAE_t3], positions=[1, 2, 3, 4], widths=0.3, patch_artist=True,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops)

# Overlay scatter plots on top of the box plots
colors = ['indianred', 'darkorange', 'seagreen', 'royalblue']
scatter_sizes = [sizes_t0, sizes_t1, sizes_t2, sizes_t3]  # List of size arrays for legend

for i, (RAE_data, x_val, color) in enumerate(zip([RAE_t0, RAE_t1, RAE_t2, RAE_t3], [1, 2, 3, 4], colors)):
    plt.scatter(x_val + np.random.normal(0, 0.2, len(RAE_data)), RAE_data, 
                s=np.array(scatter_sizes[i])/80, 
                color=color, alpha=0.7)

# Create a custom legend for area sizes
legend_elements = []

# Pick one random sample from each group for the legend
for sizes in scatter_sizes:
    if len(sizes) > 0:
        random_index = random.choice(range(len(sizes)))
        sample_size = sizes[random_index]
        area_km2 = round(sample_size / 10000, 2)  # Calculate area in km²
        legend_elements.append(plt.scatter([], [], s=sample_size/80, color='black', alpha=0.7, label=f"{area_km2} km²"))

# Add legend to the plot
legend = plt.legend(handles=legend_elements, title='Sample Areas', fontsize=30, title_fontsize=25, loc='upper right', bbox_to_anchor=(1.0, 1), ncol=1, frameon=False, borderpad=1, labelspacing=1.5)
legend._legend_box.align = "center"  # Align legend to the center

# Customize the plot
plt.xlabel('Groups', fontsize=30)
plt.ylabel('RAE Values', fontsize=30)
plt.xticks([1, 2, 3, 4], ['RAE_t0', 'RAE_t1', 'RAE_t2', 'RAE_t3'], fontsize=20)
plt.yticks(fontsize=30)
plt.grid(True)
plt.show()
