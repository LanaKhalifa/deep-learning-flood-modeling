#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:05:15 2024

@author: lana_k
"""

import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Base directory containing the project folders
base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/HECRAS_Simulations_Results'
# Project folder names
project_folders = ['prj_03', 'prj_04', 'prj_05', 'prj_06']

# List to store areas of all TIFF files
areas = []

# Function to calculate area of a single TIFF file
def calculate_area(tif_file):
    print('im here')
    with rasterio.open(tif_file) as src:
        # Get number of pixels
        num_pixels = src.width * src.height
        print('im here')
        print('src.width', src.width)
        print('src.height', src.height)
        
        # Calculate area (assuming pixel size is 1 square unit)
        area = 1 * num_pixels
        return area

# Iterate through all project folders and their TIFF files
for project in project_folders:
    terrain_dir = os.path.join(base_dir, project, 'Terrains')
    print(terrain_dir)
    # Find all TIFF files in the current Terrains folder
    tif_files = glob.glob(os.path.join(terrain_dir, '*.tif'))
    print('here', tif_files)
    
    for tif_file in tif_files:
        print(tif_file)
        area = calculate_area(tif_file)
        areas.append(area)

# Calculate mean and standard deviation of the areas
mean_area = np.mean(areas)
std_dev_area = np.std(areas)
max_area = np.max(areas)

# Print results
print(f"Mean Area: {mean_area}")
print(f"Standard Deviation of Area: {std_dev_area}")

# Plotting the box plot for all areas
plt.figure(figsize=(10, 6))
plt.boxplot(areas, vert=False)
plt.title('Box Plot of Areas of TIFF Files')
plt.xlabel('Area ($10^6$)')
plt.grid(True)

# Set x-axis formatter to scientific notation with 10^6 as base
formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((6, 6))
plt.gca().xaxis.set_major_formatter(formatter)
plt.show()
