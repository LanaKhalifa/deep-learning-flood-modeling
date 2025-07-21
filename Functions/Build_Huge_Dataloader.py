import pickle
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Functions')
from zero_internal import zero_internal
from plot_samples import plot_samples

# Define file paths
base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Database/'
save_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Dataloaders/'

boundary_thickness = 2
BC = 2
batch_size = 300
num_cells_in_patch = 32
#%% big_train_val
# Load the data from pickle files
with open(os.path.join(base_dir, 'big_train_val_depths_preprocessed.pkl'), 'rb') as f:
    depths_numpy = np.array(pickle.load(f))
with open(os.path.join(base_dir, 'big_train_val_depths_next_preprocessed.pkl'), 'rb') as f:
    depths_next_numpy = np.array(pickle.load(f))
with open(os.path.join(base_dir, 'big_train_val_terrains_preprocessed.pkl'), 'rb') as f:
    terrains_numpy = np.array(pickle.load(f))

# Plot samples
num_samples = 20 # Number of samples to plot
indices = np.random.choice(len(depths_numpy), num_samples, replace=False) # Randomly pick indices
fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
for i, idx in enumerate(indices):
    # Plot terrains_numpy
    axes[i, 0].imshow(terrains_numpy[idx], cmap='viridis')
    axes[i, 0].set_title(f'terrains_numpy[{idx}]')
    axes[i, 0].axis('off')
    # Plot depths_numpy
    axes[i, 1].imshow(depths_numpy[idx], cmap='viridis')
    axes[i, 1].set_title(f'depths_numpy[{idx}]')
    axes[i, 1].axis('off')
    # Plot depths_next_numpy
    axes[i, 2].imshow(depths_next_numpy[idx], cmap='viridis')
    axes[i, 2].set_title(f'depths_next_numpy[{idx}]')
    axes[i, 2].axis('off')
plt.tight_layout()
plt.show()

# Create Data
N = terrains_numpy.shape[0]
terrains_expanded = np.expand_dims(terrains_numpy, axis=1)
terrains_tensor = torch.tensor(terrains_expanded).double()
dataset_numpy = np.zeros((N, 2, num_cells_in_patch, num_cells_in_patch)) # Prepare dataset
depths_for_zeroing = copy.deepcopy(depths_next_numpy)
depths_BC_numpy = [zero_internal(matrix, boundary_thickness) for matrix in depths_for_zeroing]
depths_BC_numpy = np.stack(depths_BC_numpy, axis=0)
for i in range(N):
    stacked = np.stack((depths_BC_numpy[i], depths_numpy[i]))
    dataset_numpy[i] = stacked
labels_numpy = np.expand_dims(depths_next_numpy - depths_numpy, axis=1)
labels_tensor = torch.tensor(labels_numpy).double()
dataset_tensor = torch.tensor(dataset_numpy).double()

# Shuffle and split the dataset
indices = torch.randperm(dataset_tensor.size(0))
dataset_tensor = dataset_tensor[indices]
labels_tensor = labels_tensor[indices]
cle = terrains_tensor[indices]
train_size = int(N * 0.7)
train_terrains = terrains_tensor[:train_size]
train_data = dataset_tensor[:train_size]
train_labels = labels_tensor[:train_size]
val_terrains = terrains_tensor[train_size:]
val_data = dataset_tensor[train_size:]
val_labels = labels_tensor[train_size:]

# Create TensorDatasets
train_dataset = TensorDataset(train_terrains, train_data, train_labels)
val_dataset = TensorDataset(val_terrains, val_data, val_labels)
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Plot samples
plot_samples(train_loader, samples_to_plot=100)
# Save DataLoaders
torch.save(train_loader, os.path.join(save_dir, f'big_train_loader_Terrain_1_BC_{boundary_thickness}.pt'))
torch.save(val_loader, os.path.join(save_dir, f'big_val_loader_Terrain_1_BC_{boundary_thickness}.pt'))
print('length of train_loader', len(train_loader))
print("Data loaders saved successfully.")
#%% big_test
# Load the data from pickle files
with open(os.path.join(base_dir, 'big_test_depths_preprocessed.pkl'), 'rb') as f:
    depths_numpy = np.array(pickle.load(f))
with open(os.path.join(base_dir, 'big_test_depths_next_preprocessed.pkl'), 'rb') as f:
    depths_next_numpy = np.array(pickle.load(f))
with open(os.path.join(base_dir, 'big_test_terrains_preprocessed.pkl'), 'rb') as f:
    terrains_numpy = np.array(pickle.load(f))

# Plot samples
num_samples = 20 # Number of samples to plot
indices = np.random.choice(len(depths_numpy), num_samples, replace=False) # Randomly pick indices
fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
for i, idx in enumerate(indices):
    # Plot terrains_numpy
    axes[i, 0].imshow(terrains_numpy[idx], cmap='viridis')
    axes[i, 0].set_title(f'terrains_numpy[{idx}]')
    axes[i, 0].axis('off')
    # Plot depths_numpy
    axes[i, 1].imshow(depths_numpy[idx], cmap='viridis')
    axes[i, 1].set_title(f'depths_numpy[{idx}]')
    axes[i, 1].axis('off')
    # Plot depths_next_numpy
    axes[i, 2].imshow(depths_next_numpy[idx], cmap='viridis')
    axes[i, 2].set_title(f'depths_next_numpy[{idx}]')
    axes[i, 2].axis('off')
plt.tight_layout()
plt.show()

# Create Data
N = terrains_numpy.shape[0]
terrains_expanded = np.expand_dims(terrains_numpy, axis=1)
terrains_tensor = torch.tensor(terrains_expanded).double()
dataset_numpy = np.zeros((N, 2, num_cells_in_patch, num_cells_in_patch)) # Prepare dataset
depths_for_zeroing = copy.deepcopy(depths_next_numpy)
depths_BC_numpy = [zero_internal(matrix, boundary_thickness) for matrix in depths_for_zeroing]
depths_BC_numpy = np.stack(depths_BC_numpy, axis=0)
for i in range(N):
    stacked = np.stack((depths_BC_numpy[i], depths_numpy[i]))
    dataset_numpy[i] = stacked
labels_numpy = np.expand_dims(depths_next_numpy - depths_numpy, axis=1)
labels_tensor = torch.tensor(labels_numpy).double()
dataset_tensor = torch.tensor(dataset_numpy).double()

# Shuffle and split the dataset
indices = torch.randperm(dataset_tensor.size(0))
dataset_tensor = dataset_tensor[indices]
labels_tensor = labels_tensor[indices]
cle = terrains_tensor[indices]
train_size = int(N * 0.7)
train_terrains = terrains_tensor[:train_size]
train_data = dataset_tensor[:train_size]
train_labels = labels_tensor[:train_size]
val_terrains = terrains_tensor[train_size:]
val_data = dataset_tensor[train_size:]
val_labels = labels_tensor[train_size:]

# Create TensorDatasets
train_dataset = TensorDataset(train_terrains, train_data, train_labels)
val_dataset = TensorDataset(val_terrains, val_data, val_labels)
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Plot samples
plot_samples(train_loader, samples_to_plot=100)
# Save DataLoaders
torch.save(train_loader, os.path.join(save_dir, f'big_test_loader_Terrain_1_BC_{boundary_thickness}.pt'))
torch.save(val_loader, os.path.join(save_dir, f'big_test_loader_Terrain_1_BC_{boundary_thickness}.pt'))
print('length of train_loader', len(train_loader))
print("Data loaders saved successfully.")
#%% small_train_val
# Load the data from pickle files
with open(os.path.join(base_dir, 'small_train_val_depths_preprocessed.pkl'), 'rb') as f:
    depths_numpy = np.array(pickle.load(f))
with open(os.path.join(base_dir, 'small_train_val_depths_next_preprocessed.pkl'), 'rb') as f:
    depths_next_numpy = np.array(pickle.load(f))
with open(os.path.join(base_dir, 'small_train_val_terrains_preprocessed.pkl'), 'rb') as f:
    terrains_numpy = np.array(pickle.load(f))

# Plot samples
num_samples = 20 # Number of samples to plot
indices = np.random.choice(len(depths_numpy), num_samples, replace=False) # Randomly pick indices
fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
for i, idx in enumerate(indices):
    # Plot terrains_numpy
    axes[i, 0].imshow(terrains_numpy[idx], cmap='viridis')
    axes[i, 0].set_title(f'terrains_numpy[{idx}]')
    axes[i, 0].axis('off')
    # Plot depths_numpy
    axes[i, 1].imshow(depths_numpy[idx], cmap='viridis')
    axes[i, 1].set_title(f'depths_numpy[{idx}]')
    axes[i, 1].axis('off')
    # Plot depths_next_numpy
    axes[i, 2].imshow(depths_next_numpy[idx], cmap='viridis')
    axes[i, 2].set_title(f'depths_next_numpy[{idx}]')
    axes[i, 2].axis('off')
plt.tight_layout()
plt.show()

# Create Data
N = terrains_numpy.shape[0]
terrains_expanded = np.expand_dims(terrains_numpy, axis=1)
terrains_tensor = torch.tensor(terrains_expanded).double()
dataset_numpy = np.zeros((N, 2, num_cells_in_patch, num_cells_in_patch)) # Prepare dataset
depths_for_zeroing = copy.deepcopy(depths_next_numpy)
depths_BC_numpy = [zero_internal(matrix, boundary_thickness) for matrix in depths_for_zeroing]
depths_BC_numpy = np.stack(depths_BC_numpy, axis=0)
for i in range(N):
    stacked = np.stack((depths_BC_numpy[i], depths_numpy[i]))
    dataset_numpy[i] = stacked
labels_numpy = np.expand_dims(depths_next_numpy - depths_numpy, axis=1)
labels_tensor = torch.tensor(labels_numpy).double()
dataset_tensor = torch.tensor(dataset_numpy).double()

# Shuffle and split the dataset
indices = torch.randperm(dataset_tensor.size(0))
dataset_tensor = dataset_tensor[indices]
labels_tensor = labels_tensor[indices]
cle = terrains_tensor[indices]
train_size = int(N * 0.7)
train_terrains = terrains_tensor[:train_size]
train_data = dataset_tensor[:train_size]
train_labels = labels_tensor[:train_size]
val_terrains = terrains_tensor[train_size:]
val_data = dataset_tensor[train_size:]
val_labels = labels_tensor[train_size:]

# Create TensorDatasets
train_dataset = TensorDataset(train_terrains, train_data, train_labels)
val_dataset = TensorDataset(val_terrains, val_data, val_labels)
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Plot samples
plot_samples(train_loader, samples_to_plot=100)
# Save DataLoaders
torch.save(train_loader, os.path.join(save_dir, f'small_train_loader_Terrain_1_BC_{boundary_thickness}.pt'))
torch.save(val_loader, os.path.join(save_dir, f'small_val_loader_Terrain_1_BC_{boundary_thickness}.pt'))
print('length of train_loader', len(train_loader))
print("Data loaders saved successfully.")