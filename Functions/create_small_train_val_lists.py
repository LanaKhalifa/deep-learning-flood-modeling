import pickle
import os
import numpy as np
#%% prj 03
prj_num = '03'
directory = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Database/{prj_num}/'
save_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Database/'
def concatenate_and_save(file_prefix, output_filename):
    all_items = []

    # Loop through the specified range of pickle files
    for i in range(3):  # only 7*3 = 21 sims
        file_path = os.path.join(directory, f'{file_prefix}_{i}.pkl')
        with open(file_path, 'rb') as file:
            items = pickle.load(file)
            # Convert each item to float32 and append to the list
            for item in items:
                if isinstance(item, np.ndarray):
                    all_items.append(item.astype(np.float32))
                else:
                    all_items.append(item)

    # Save all_items to a new pickle file
    output_file_path = os.path.join(save_dir, output_filename)
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(all_items, output_file)

    print(f"All {file_prefix} items have been saved to {output_file_path}")
    return all_items

# Concatenate and save depths, depths_next, and terrains
all_depths = concatenate_and_save('depths', 'small_train_val_depths.pkl')
all_depths_next = concatenate_and_save('depths_next', 'small_train_val_depths_next.pkl')
all_terrains = concatenate_and_save('terrains', 'small_train_val_terrains.pkl')

# Ensure the lengths of the three lists are identical
assert len(all_depths) == len(all_depths_next) == len(all_terrains), "The lengths of the lists are not identical."

# Ensure every element in all_depths is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths), "Not all items in all_depths are 32 by 32."

# Ensure every element in all_depths_next is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths_next), "Not all items in all_depths_next are 32 by 32."

# Ensure every element in all_terrains is 321 by 321
assert all(isinstance(item, np.ndarray) and item.shape == (321, 321) for item in all_terrains), "Not all items in all_terrains are 321 by 321."

print("All checks passed successfully for prj 03.")

import matplotlib.pyplot as plt
import random

#plot
# Number of samples to plot
num_samples = 20

# Randomly select 20 indices from the available data
random_indices = random.sample(range(len(all_depths)), num_samples)

# Plot the selected samples
for i, idx in enumerate(random_indices):
    # Extract the corresponding depth, depth_next, and terrain
    depth = all_depths[idx]
    depth_next = all_depths_next[idx]
    terrain = all_terrains[idx]
    
    # Calculate the depth difference
    depth_diff = depth_next - depth
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot depth
    axs[0].imshow(depth, cmap='viridis')
    axs[0].set_title(f'Sample {i+1}: Depth')
    
    # Plot depth difference
    axs[1].imshow(depth_diff, cmap='coolwarm')
    axs[1].set_title(f'Sample {i+1}: Depth Difference')
    
    # Plot terrain
    axs[2].imshow(terrain, cmap='terrain')
    axs[2].set_title(f'Sample {i+1}: Terrain')
    
    # Display the plots
    plt.tight_layout()
    plt.show()