import pickle
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

sys.path.append('/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Functions')
from zero_internal import zero_internal
from plot_samples import plot_samples
from compute_gradients import compute_gradients

# Define file paths
base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Database/02_and_03/'
save_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Dataloaders/02_and_03/'

# Load the data from pickle files
with open(os.path.join(base_dir, 'depths.pkl'), 'rb') as f:
    depths_numpy = np.array(pickle.load(f))

with open(os.path.join(base_dir, 'depths_next.pkl'), 'rb') as f:
    depths_next_numpy = np.array(pickle.load(f))

with open(os.path.join(base_dir, 'terrains.pkl'), 'rb') as f:
    terrains_numpy = np.array(pickle.load(f))

BC = 2

# Define constants
batch_size = 100
N = terrains_numpy.shape[0]
boundary_thickness = BC

# Expand terrains and convert to tensor
terrains_expanded = np.expand_dims(terrains_numpy, axis=1)
terrains_tensor = torch.tensor(terrains_expanded).double()

if True:
    cell_size = 1.0  # Example cell size

    # Compute gradient maps
    gradients_x_tensor, gradients_y_tensor = compute_gradients(terrains_tensor, cell_size)

    # Concatenate along the channel dimension
    enhanced_terrains_tensor = torch.cat([terrains_tensor, gradients_x_tensor, gradients_y_tensor], dim=1)

# Prepare dataset
num_cells_in_patch = depths_numpy.shape[-1]
dataset_numpy = np.zeros((N, 2, num_cells_in_patch, num_cells_in_patch))

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
terrains_tensor = enhanced_terrains_tensor[indices]

train_size = int(N * 0.7)
val_size = int(N * 0.2)
test_size = int(N * 0.1)

train_terrains = terrains_tensor[:train_size]
train_data = dataset_tensor[:train_size]
train_labels = labels_tensor[:train_size]

val_terrains = terrains_tensor[train_size: train_size + val_size]
val_data = dataset_tensor[train_size: train_size + val_size]
val_labels = labels_tensor[train_size: train_size + val_size]

test_terrains = terrains_tensor[-test_size:]
test_data = dataset_tensor[-test_size:]
test_labels = labels_tensor[-test_size:]

# Create TensorDatasets
train_dataset = TensorDataset(train_terrains, train_data, train_labels)
val_dataset = TensorDataset(val_terrains, val_data, val_labels)
test_dataset = TensorDataset(test_terrains, test_data, test_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Plot samples
#plot_samples(train_loader, samples_to_plot=10)

# Save DataLoaders
torch.save(train_loader, os.path.join(save_dir, f'train_loader_Terrain_3_BC_{boundary_thickness}.pt'))
torch.save(val_loader, os.path.join(save_dir, f'val_loader_Terrain_3_BC_{boundary_thickness}.pt'))
torch.save(test_loader, os.path.join(save_dir, f'test_loader_Terrain_3_BC_{boundary_thickness}.pt'))

print("Data loaders saved successfully.")
