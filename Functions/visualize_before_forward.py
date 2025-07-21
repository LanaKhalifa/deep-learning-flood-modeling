#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 01:25:10 2024

@author: lana_k
"""

import matplotlib.pyplot as plt

def visualize_before_forward(epoch, idx, terrains, data):
    """
    Visualize the terrains and data before entering the network.
    
    Parameters:
    - epoch: Current epoch number.
    - idx: Index of the current batch.
    - terrains: Terrain data.
    - data: Input data containing boundary conditions and water depth.
    """
    if epoch == 0 and idx == 0:  # Only visualize for the first batch in each epoch
        sample_terrain = terrains[0, 0].cpu().numpy()  # Extract the first sample's terrain
        sample_boundary_condition = data[0, 0].cpu().numpy()  # Extract the first sample's boundary condition
        sample_water_depth = data[0, 1].cpu().numpy()  # Extract the first sample's water depth
    
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(sample_terrain, cmap='viridis')
        axs[0].set_title('Terrain')
        axs[1].imshow(sample_boundary_condition, cmap='viridis')
        axs[1].set_title('Boundary Condition')
        axs[2].imshow(sample_water_depth, cmap='viridis')
        axs[2].set_title('Water Depth at Time n')
        plt.show()