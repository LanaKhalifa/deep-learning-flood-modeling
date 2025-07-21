import torch
import numpy as np
import matplotlib.pyplot as plt 

import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def calculate_rae_all_quartiles(netG, loader, device, path):
    netG.eval()
    raes_all_samples = []
    
    with torch.no_grad():
        for terrains, data, labels in loader:
            terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)
            
            # Model prediction
            y_fake = netG(terrains, data)
            
            # Calculate the absolute differences
            diff = torch.abs(labels - y_fake)            
            diff_sum = torch.sum(diff, dim=(1, 2, 3))
            
            # Calculate the absolute dummy differences
            dummy_diff = torch.abs(labels)
            dummy_diff_sum = torch.sum(dummy_diff, dim=(1, 2, 3))
            
            # Calculate RAE for the batch
            raes_batch = diff_sum / dummy_diff_sum
            raes_all_samples.extend(raes_batch.cpu().numpy())  # Convert to list of numbers and extend the main list
    
    # Convert to a NumPy array
    rae_array = np.array(raes_all_samples)
    
    # Filter out any infinite values
    rae_array = rae_array[np.isfinite(rae_array)]
    
    # Check for inf values
    if np.any(np.isinf(rae_array)):
        print("There are inf values in raes_all_samples.")
    else:
        print("No inf values in raes_all_samples.")
    
    # Calculate the necessary quartiles for the boxplot
    Q1 = np.percentile(rae_array, 25)
    Q2 = np.percentile(rae_array, 50)  # Median
    Q3 = np.percentile(rae_array, 75)
    
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out the outliers
    filtered_rae_array = rae_array[(rae_array >= lower_bound) & (rae_array <= upper_bound)]
    
    # Calculate the minimum and maximum values after removing outliers
    min_val = np.min(filtered_rae_array)
    max_val = np.max(filtered_rae_array)

    # Plot the box plot of RAE values
    fig, ax = plt.subplots(figsize=(1, 10))  # Adjusted figure size for better readability

    # Creating the boxplot (vert=True for vertical orientation)
    ax.boxplot([rae_array], vert=True, patch_artist=True, showmeans=False,
               showfliers=False,  # Omit outliers
               medianprops=dict(color='blue', linewidth=2),  # Blue line for median
               boxprops=dict(color='black', linewidth=2, facecolor='lightgrey'),  # Light grey box with thinner black border
               whiskerprops=dict(color='black', linewidth=2),  # Thinner whisker lines
               capprops=dict(color='black', linewidth=2),     # Thinner caps
               flierprops=dict(marker='o', color='red', alpha=0.5),  # Style for outliers
               widths=0.5)  # Keep the boxes at the same width

    # Customize tick size and position
    ax.tick_params(axis='y', which='major', labelsize=20, labelright=True, labelleft=False)  # Place y-axis ticks on the right

    # Customize the background grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust the layout
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)

    # Save the plot to the specified path, ensuring no cropping occurs
    plt.savefig(path, bbox_inches='tight', dpi=300)  # Added bbox_inches='tight'

    # Show the plot
    plt.show()

    return Q1, Q2, Q3, min_val, max_val

def calculate_rae_all_quartiles_old(netG, loader, device, path):
    netG.eval()
    raes_all_samples = []
    
    with torch.no_grad():
        for terrains, data, labels in loader:
            terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)
            
            # Model prediction
            y_fake = netG(terrains, data)
            
            # Calculate the absolute differences
            diff = torch.abs(labels - y_fake)            
            diff_sum = torch.sum(diff, dim=(1, 2, 3))
            
            # Calculate the absolute dummy differences
            dummy_diff = torch.abs(labels)
            dummy_diff_sum = torch.sum(dummy_diff, dim=(1, 2, 3))
            
            # Calculate RAE for the batch
            raes_batch = diff_sum / dummy_diff_sum
            raes_all_samples.extend(raes_batch.cpu().numpy())  # Convert to list of numbers and extend the main list
    
    # Convert to a NumPy array
    rae_array = np.array(raes_all_samples)
    
    # Filter out any infinite values
    rae_array = rae_array[np.isfinite(rae_array)]
    
    # Check for inf values
    if np.any(np.isinf(rae_array)):
        print("There are inf values in raes_all_samples.")
    else:
        print("No inf values in raes_all_samples.")
    
    # Calculate the necessary quartiles for the boxplot
    Q1 = np.percentile(rae_array, 25)
    Q2 = np.percentile(rae_array, 50)  # Median
    Q3 = np.percentile(rae_array, 75)
    
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out the outliers
    filtered_rae_array = rae_array[(rae_array >= lower_bound) & (rae_array <= upper_bound)]
    
    # Calculate the minimum and maximum values after removing outliers
    min_val = np.min(filtered_rae_array)
    max_val = np.max(filtered_rae_array)

    # Plot the box plot of RAE values
    fig, ax = plt.subplots(figsize=(3, 10))  # Adjusted figure size for better readability

    # Creating the boxplot (vert=True for vertical orientation)
    ax.boxplot([rae_array], vert=True, patch_artist=True, showmeans=False,
               showfliers=False,  # Omit outliers
               medianprops=dict(color='blue', linewidth=2),  # Blue line for median
               boxprops=dict(color='black', linewidth=2, facecolor='lightgrey'),  # Light grey box with thinner black border
               whiskerprops=dict(color='black', linewidth=2),  # Thinner whisker lines
               capprops=dict(color='black', linewidth=2),     # Thinner caps
               flierprops=dict(marker='o', color='red', alpha=0.5),  # Style for outliers
               widths=0.2)  # Keep the boxes at the same width

    # Set the y-axis label
    #ax.set_ylabel('Relative Absolute Error (RAE)', fontsize=20)

    # Customize the x-axis ticks
    ax.set_xticks([1])
    #ax.set_xticklabels(['prj_01_test'], fontsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20, labelright=True, labelleft=False)  # Place y-axis ticks on the right


    # Customize the background grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust the layout
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)

    # Save the plot to the specified path
    plt.savefig(path, bbox_inches='tight')

    # Show the plot
    plt.show()

    return Q1, Q2, Q3, min_val, max_val


                
