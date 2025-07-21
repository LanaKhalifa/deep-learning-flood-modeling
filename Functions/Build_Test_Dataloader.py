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

# Define file paths
base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Database/'
save_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Dataloaders/'

# Load the data from pickle files
with open(os.path.join(base_dir, 'test_depths_filtered.pkl'), 'rb') as f:
    depths_numpy = np.array(pickle.load(f))

with open(os.path.join(base_dir, 'test_depths_next_filtered.pkl'), 'rb') as f:
    depths_next_numpy = np.array(pickle.load(f))

with open(os.path.join(base_dir, 'test_terrains_filtered.pkl'), 'rb') as f:
    terrains_numpy = np.array(pickle.load(f))

BC = 2

# Define constants
batch_size = 100
N = terrains_numpy.shape[0]
boundary_thickness = BC

# Expand terrains and convert to tensor
terrains_expanded = np.expand_dims(terrains_numpy, axis=1)
terrains_tensor = torch.tensor(terrains_expanded).double()

# Prepare dataset
num_cells_in_patch = 32
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
terrains_tensor = terrains_tensor[indices]

test_size = N


test_terrains = terrains_tensor[:]
test_data = dataset_tensor[:]
test_labels = labels_tensor[:]


# Create TensorDatasets
test_dataset = TensorDataset(test_terrains, test_data, test_labels)

# Create DataLoaders
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Plot samples
plot_samples(test_loader, samples_to_plot=100)


if False:
    # Save DataLoaders
    torch.save(test_loader, os.path.join(save_dir, f'test_loader_Terrain_1_BC_{boundary_thickness}.pt'))
    
    print("Data loaders saved successfully.")
