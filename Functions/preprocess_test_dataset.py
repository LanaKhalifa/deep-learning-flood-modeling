import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

output_directory = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Database/'

# Load the lists
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

huge_depths = load_pickle(os.path.join(output_directory, 'test_depths.pkl'))
huge_depths_next = load_pickle(os.path.join(output_directory, 'test_depths_next.pkl'))
huge_terrains = load_pickle(os.path.join(output_directory, 'test_terrains.pkl'))

#%% convert to np arrays 
huge_depths = np.array(huge_depths)
huge_depths_next = np.array(huge_depths_next)
huge_terrains = np.array(huge_terrains)

huge_depths_diff = huge_depths_next - huge_depths


#%%

# Select one sample for demonstration
sample_index = 0
sample_before_shift = huge_terrains[sample_index].copy()

# Shift the selected terrain sample towards zero
huge_terrains[sample_index] -= np.min(huge_terrains[sample_index])

# Plot the selected terrain sample before and after shifting
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Elevations of Terrain Sample Before and After Shifting', fontsize=30)

# Before shifting
ax = axes[0]
im = ax.imshow(sample_before_shift, cmap='terrain')
ax.set_title('Before Shifting Elevations [meters]', fontsize=30)
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=30)

# After shifting
ax = axes[1]
im = ax.imshow(huge_terrains[sample_index], cmap='terrain')
ax.set_title('After Shifting Elevations [meters]', fontsize=30)
ax.axis('off')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=30)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.4)  # Increase space between plots
plt.show()



#%% Mission a: Calculate huge_depths_diff and create a histogram of depth diff > 10^-5
plt.hist(huge_depths_diff.flatten(), bins=100)
plt.title('Histogram of all pixels in huge_depths_diff')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

#%% Mission b: Identify and remove samples with extreme pixel values
mask = np.any((huge_depths_diff > 1) | (huge_depths_diff < -1), axis=(1, 2))
# Ensure mask is a boolean array
# Count the number of elements which are True in the mask (i.e., samples with extreme values)
num_true_elements = np.sum(mask)
print(f"Number of samples with extreme values: {num_true_elements}")

mask = ~mask

num_true_elements = np.sum(mask)
print(f"Number of samples with valid values: {num_true_elements}")

huge_depths = huge_depths[mask]
huge_depths_next = huge_depths_next[mask]
huge_terrains = huge_terrains[mask]

#%% Mission c: Realculate huge_depths_diff and create a histogram of depth diff > 10^-5
huge_depths_diff = huge_depths_next - huge_depths 
plt.hist(huge_depths_diff.flatten(), bins=100)
plt.title('Histogram of all pixels in huge_depths_diff')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
    
#%% Mission d: Create a histogram for Terrains
plt.hist(huge_terrains.flatten(), bins=100)
plt.title('Histogram of all pixels in Terrains Before Shifting')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

#%% Mission e: shift all terrains towards zero
huge_terrains -= np.min(huge_terrains, axis=(1, 2), keepdims=True)

#%% Mission f: Create a histogram for Terrains after shifting
plt.hist(huge_terrains.flatten(), bins=100)
plt.title('Histogram of all pixels in Terrains After Shifting')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()

#%% Mission i: Identify and remove samples from terrain if a pixel is  > 100m
mask = np.any((huge_terrains < 100), axis=(1, 2))

huge_depths = huge_depths[mask]
huge_depths_next = huge_depths_next[mask]
huge_terrains = huge_terrains[mask]

#%% save 
# Define the base directory
base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Database/'

# Save the arrays as pickle files
def save_pickle(array, filename):
    file_path = os.path.join(base_dir, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(array, file)

#%% # Save the filtered arrays
save_pickle(huge_depths, 'test_depths_filtered.pkl')
save_pickle(huge_depths_next, 'test_depths_next_filtered.pkl')
save_pickle(huge_terrains, 'test_terrains_filtered.pkl')

print("Filtered arrays have been saved successfully.")

