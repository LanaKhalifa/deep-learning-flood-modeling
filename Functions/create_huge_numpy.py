import pickle
import os

# List of project numbers
project_numbers = ['03', '04', '05', '06']

# Initialize lists to hold the concatenated results
huge_depths = []
huge_depths_next = []
huge_terrains = []

# Function to load and concatenate data from pickle files
def load_and_concatenate(prj_num, file_name, huge_list):
    directory = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Database/{prj_num}/'
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        huge_list.extend(data)
    return huge_list

# Load and concatenate data from all project directories
for prj_num in project_numbers:
    huge_depths = load_and_concatenate(prj_num, 'all_depths.pkl', huge_depths)
    huge_depths_next = load_and_concatenate(prj_num, 'all_depths_next.pkl', huge_depths_next)
    huge_terrains = load_and_concatenate(prj_num, 'all_terrains.pkl', huge_terrains)

if True:
    # Save the concatenated lists to new pickle files
    output_directory = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Database/'
    
    with open(os.path.join(output_directory, 'huge_depths.pkl'), 'wb') as output_file:
        pickle.dump(huge_depths, output_file)
    
    with open(os.path.join(output_directory, 'huge_depths_next.pkl'), 'wb') as output_file:
        pickle.dump(huge_depths_next, output_file)
    
    with open(os.path.join(output_directory, 'huge_terrains.pkl'), 'wb') as output_file:
        pickle.dump(huge_terrains, output_file)
    
    print("All huge lists have been saved successfully.")

#%%
import numpy as np
import matplotlib.pyplot as plt


# Number of samples to plot
num_samples = 100

# Randomly pick indices
indices = np.random.choice(len(huge_depths), num_samples, replace=False)

# Plot the samples
fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))

for i, idx in enumerate(indices):
    # Plot huge_depths
    axes[i, 0].imshow(huge_depths[idx], cmap='viridis')
    axes[i, 0].set_title(f'huge_depths[{idx}]')
    axes[i, 0].axis('off')
    
    # Plot huge_depths_next
    axes[i, 1].imshow(huge_depths_next[idx], cmap='viridis')
    axes[i, 1].set_title(f'huge_depths_next[{idx}]')
    axes[i, 1].axis('off')
    
    # Plot huge_terrains
    axes[i, 2].imshow(huge_terrains[idx], cmap='viridis')
    axes[i, 2].set_title(f'huge_terrains[{idx}]')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()

