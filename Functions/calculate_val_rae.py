import torch
import numpy as np
import matplotlib.pyplot as plt 

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
            
            # Calculate the absolute true differences
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
    
    # Calculate the minimum and maximum values
    min_val = np.min(rae_array)
    max_val = np.max(rae_array)

    # Save the box plot values (without outliers)
    boxplot_values = {
        'min': min_val,
        'Q1': Q1,
        'median': Q2,
        'Q3': Q3,
        'max': max_val
    }
  
    # Plot the box plot of RAE values with the same style as the provided code
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Creating the boxplot
    ax.boxplot([rae_array], vert=False, patch_artist=True, showmeans=False,
               medianprops=dict(color='blue', linewidth=2),  # Blue line for median
               boxprops=dict(color='black', linewidth=2, facecolor='lightgrey'),  # Light grey box with thinner black border
               whiskerprops=dict(color='black', linewidth=2),  # Thinner whisker lines
               capprops=dict(color='black', linewidth=2),     # Thinner caps
               flierprops=dict(marker='o', color='red', alpha=0.5),  # Style for outliers
               widths=0.5)  # Keep the boxes at the same width
    
    # Set the x-axis label and title
    ax.set_xlabel('Relative Absolute Error (RAE)', fontsize=16)
    ax.set_title('Box Plot of RAE Values for Validation Set', fontsize=18)  # Optional: Add bold fontweight
    
    # Customize the background grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust the layout
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)
    
    # Save the plot to the specified path
    plt.savefig(path, dpi=300)
    
    # Show the plot
    plt.show()    
    
    return boxplot_values
    

