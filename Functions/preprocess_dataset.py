import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Database/'

# Save the arrays as pickle files
def save_pickle(array, filename):
    file_path = os.path.join(base_dir, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(array, file)
        
# Load the lists
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

#%% big_train_val
huge_depths = load_pickle(os.path.join(base_dir, 'big_train_val_depths.pkl'))
huge_depths_next = load_pickle(os.path.join(base_dir, 'big_train_val_depths_next.pkl'))
huge_terrains = load_pickle(os.path.join(base_dir, 'big_train_val_terrains.pkl'))

# convert to np arraysand count
huge_depths = np.array(huge_depths)
huge_depths_next = np.array(huge_depths_next)
huge_terrains = np.array(huge_terrains)
huge_depths_diff = huge_depths_next - huge_depths
print("Number Total Samples: ", len(huge_depths))

# Plot histogram before filtering
plt.figure(figsize=(10, 6))
plt.hist(huge_depths_diff.flatten(), bins=100, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of Depth Differences (Before Filtering)')
plt.grid(True)
plt.show()

# Mission a: Identify and remove samples with extreme pixel values
mask = np.any((huge_depths_diff > 10), axis=(1, 2))
num_true_elements = np.sum(mask)
mask = ~mask
num_true_elements = np.sum(mask)
huge_depths = huge_depths[mask]
huge_depths_next = huge_depths_next[mask]
huge_terrains = huge_terrains[mask]
print("Number Total Samples after filtering Depths: ", len(huge_depths))

# Plot histogram after filtering
plt.figure(figsize=(10, 6))
plt.hist(huge_terrains.flatten(), bins=100, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram of Depth Differences (After Filtering)')
plt.grid(True)
plt.show()

# plot and save a histogram of water depth diff pixels after flattenning all samples:
# Plot and save a histogram of water depth diff pixels after flattening all samples:
huge_depths_diff_filtered = huge_depths_next - huge_depths  # Recalculate the differences after filtering
flattened_depths_diff = huge_depths_diff_filtered.flatten()  # Flatten all samples to a 1D array

plt.figure(figsize=(10, 6))
plt.hist(flattened_depths_diff, bins=100, color='blue', edgecolor='black')
plt.title('Histogram of Water Depth Differences After Filtering')
plt.xlabel('Water Depth Difference (meters)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Save the histogram as an image file
histogram_path = os.path.join(base_dir, 'big_train_val_water_depth_difference_histogram.png')
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')

# Mission b: shift all terrains towards zero
huge_terrains -= np.min(huge_terrains, axis=(1, 2), keepdims=True)

# save 
save_pickle(huge_depths,'big_train_val_depths_preprocessed.pkl')
save_pickle(huge_depths_next,'big_train_val_depths_next_preprocessed.pkl')
save_pickle(huge_terrains,'big_train_val_terrains_preprocessed.pkl')
#%% big_test
huge_depths = load_pickle(os.path.join(base_dir, 'big_test_depths.pkl'))
huge_depths_next = load_pickle(os.path.join(base_dir, 'big_test_depths_next.pkl'))
huge_terrains = load_pickle(os.path.join(base_dir, 'big_test_terrains.pkl'))

# convert to np arraysand count
huge_depths = np.array(huge_depths)
huge_depths_next = np.array(huge_depths_next)
huge_terrains = np.array(huge_terrains)
huge_depths_diff = huge_depths_next - huge_depths
print("Number Total Samples: ", len(huge_depths))

# Mission a: Identify and remove samples with extreme pixel values
mask = np.any((huge_depths_diff > 10), axis=(1, 2))
num_true_elements = np.sum(mask)
mask = ~mask
num_true_elements = np.sum(mask)
huge_depths = huge_depths[mask]
huge_depths_next = huge_depths_next[mask]
huge_terrains = huge_terrains[mask]
print("Number Total Samples after filtering Depths: ", len(huge_depths))

# plot and save a histogram of water depth diff pixels after flattenning all samples:
# Plot and save a histogram of water depth diff pixels after flattening all samples:
huge_depths_diff_filtered = huge_depths_next - huge_depths  # Recalculate the differences after filtering
flattened_depths_diff = huge_depths_diff_filtered.flatten()  # Flatten all samples to a 1D array

plt.figure(figsize=(10, 6))
plt.hist(flattened_depths_diff, bins=100, color='blue', edgecolor='black')
plt.title('Histogram of Water Depth Differences After Filtering')
plt.xlabel('Water Depth Difference (meters)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Save the histogram as an image file
histogram_path = os.path.join(base_dir, 'big_test_water_depth_difference_histogram.png')
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')

# Mission b: shift all terrains towards zero
huge_terrains -= np.min(huge_terrains, axis=(1, 2), keepdims=True)

# save 
save_pickle(huge_depths,'big_test_depths_preprocessed.pkl')
save_pickle(huge_depths_next,'big_test_depths_next_preprocessed.pkl')
save_pickle(huge_terrains,'big_test_terrains_preprocessed.pkl')

#%% small_train_val
huge_depths = load_pickle(os.path.join(base_dir, 'small_train_val_depths.pkl'))
huge_depths_next = load_pickle(os.path.join(base_dir, 'small_train_val_depths_next.pkl'))
huge_terrains = load_pickle(os.path.join(base_dir, 'small_train_val_terrains.pkl'))

# convert to np arraysand count
huge_depths = np.array(huge_depths)
huge_depths_next = np.array(huge_depths_next)
huge_terrains = np.array(huge_terrains)
huge_depths_diff = huge_depths_next - huge_depths
print("Number Total Samples: ", len(huge_depths))

# Mission a: Identify and remove samples with extreme pixel values
mask = np.any((huge_depths_diff > 10), axis=(1, 2))
num_true_elements = np.sum(mask)
mask = ~mask
num_true_elements = np.sum(mask)
huge_depths = huge_depths[mask]
huge_depths_next = huge_depths_next[mask]
huge_terrains = huge_terrains[mask]
print("Number Total Samples after filtering Depths: ", len(huge_depths))

# plot and save a histogram of water depth diff pixels after flattenning all samples:
# Plot and save a histogram of water depth diff pixels after flattening all samples:
huge_depths_diff_filtered = huge_depths_next - huge_depths  # Recalculate the differences after filtering
flattened_depths_diff = huge_depths_diff_filtered.flatten()  # Flatten all samples to a 1D array

plt.figure(figsize=(10, 6))
plt.hist(flattened_depths_diff, bins=100, color='blue', edgecolor='black')
plt.title('Histogram of Water Depth Differences After Filtering')
plt.xlabel('Water Depth Difference (meters)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Save the histogram as an image file
histogram_path = os.path.join(base_dir, 'small_train_val_water_depth_difference_histogram.png')
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')

# Mission b: shift all terrains towards zero
huge_terrains -= np.min(huge_terrains, axis=(1, 2), keepdims=True)

# save 
save_pickle(huge_depths,'small_train_val_depths_preprocessed.pkl')
save_pickle(huge_depths_next,'small_train_val_depths_next_preprocessed.pkl')
save_pickle(huge_terrains,'small_train_val_terrains_preprocessed.pkl')

#%% prj_03_train_val
prj_03_depths = load_pickle(os.path.join(base_dir, 'prj_03_train_val_depths.pkl'))
prj_03_depths_next = load_pickle(os.path.join(base_dir, 'prj_03_train_val_depths_next.pkl'))
prj_03_terrains = load_pickle(os.path.join(base_dir, 'prj_03_train_val_terrains.pkl'))

# convert to np arraysand count
prj_03_depths = np.array(prj_03_depths)
prj_03_depths_next = np.array(prj_03_depths_next)
prj_03_terrains = np.array(prj_03_terrains)
prj_03_depths_diff = prj_03_depths_next - prj_03_depths
print("Number Total Samples: ", len(prj_03_depths))

# Mission a: Identify and remove samples with extreme pixel values
mask = np.any((prj_03_depths_diff > 10), axis=(1, 2))
num_true_elements = np.sum(mask)
mask = ~mask
num_true_elements = np.sum(mask)
prj_03_depths = prj_03_depths[mask]
prj_03_depths_next = prj_03_depths_next[mask]
prj_03_terrains = prj_03_terrains[mask]
print("Number Total Samples after filtering Depths: ", len(prj_03_depths))

# plot and save a histogram of water depth diff pixels after flattenning all samples:
# Plot and save a histogram of water depth diff pixels after flattening all samples:
prj_03_depths_diff_filtered = prj_03_depths_next - prj_03_depths  # Recalculate the differences after filtering
flattened_depths_diff = prj_03_depths_diff_filtered.flatten()  # Flatten all samples to a 1D array

plt.figure(figsize=(10, 6))
plt.hist(flattened_depths_diff, bins=100, color='blue', edgecolor='black')
plt.title('Histogram of Water Depth Differences After Filtering')
plt.xlabel('Water Depth Difference (meters)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Save the histogram as an image file
histogram_path = os.path.join(base_dir, 'prj_03_train_val_water_depth_difference_histogram.png')
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')

# Mission b: shift all terrains towards zero
prj_03_terrains -= np.min(prj_03_terrains, axis=(1, 2), keepdims=True)

# save 
save_pickle(prj_03_depths,'prj_03_train_val_depths_preprocessed.pkl')
save_pickle(prj_03_depths_next,'prj_03_train_val_depths_next_preprocessed.pkl')
save_pickle(prj_03_terrains,'prj_03_train_val_terrains_preprocessed.pkl')
#%% prj_03_test
prj_03_depths = load_pickle(os.path.join(base_dir, 'prj_03_test_depths.pkl'))
prj_03_depths_next = load_pickle(os.path.join(base_dir, 'prj_03_test_depths_next.pkl'))
prj_03_terrains = load_pickle(os.path.join(base_dir, 'prj_03_test_terrains.pkl'))

# convert to np arraysand count
prj_03_depths = np.array(prj_03_depths)
prj_03_depths_next = np.array(prj_03_depths_next)
prj_03_terrains = np.array(prj_03_terrains)
prj_03_depths_diff = prj_03_depths_next - prj_03_depths
print("Number Total Samples: ", len(prj_03_depths))

# Mission a: Identify and remove samples with extreme pixel values
mask = np.any((prj_03_depths_diff > 10), axis=(1, 2))
num_true_elements = np.sum(mask)
mask = ~mask
num_true_elements = np.sum(mask)
prj_03_depths = prj_03_depths[mask]
prj_03_depths_next = prj_03_depths_next[mask]
prj_03_terrains = prj_03_terrains[mask]
print("Number Total Samples after filtering Depths: ", len(prj_03_depths))

# plot and save a histogram of water depth diff pixels after flattenning all samples:
# Plot and save a histogram of water depth diff pixels after flattening all samples:
prj_03_depths_diff_filtered = prj_03_depths_next - prj_03_depths  # Recalculate the differences after filtering
flattened_depths_diff = prj_03_depths_diff_filtered.flatten()  # Flatten all samples to a 1D array

plt.figure(figsize=(10, 6))
plt.hist(flattened_depths_diff, bins=100, color='blue', edgecolor='black')
plt.title('Histogram of Water Depth Differences After Filtering')
plt.xlabel('Water Depth Difference (meters)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
# Save the histogram as an image file
histogram_path = os.path.join(base_dir, 'prj_03_test_water_depth_difference_histogram.png')
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')

# Mission b: shift all terrains towards zero
prj_03_terrains -= np.min(prj_03_terrains, axis=(1, 2), keepdims=True)

# save 
save_pickle(prj_03_depths,'prj_03_test_depths_preprocessed.pkl')
save_pickle(prj_03_depths_next,'prj_03_test_depths_next_preprocessed.pkl')
save_pickle(prj_03_terrains,'prj_03_test_terrains_preprocessed.pkl')
