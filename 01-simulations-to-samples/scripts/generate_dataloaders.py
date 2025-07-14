import os
import pickle
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import copy

def zero_internal(matrix, boundary_thickness):
    """Zero out the internal region of a 2D matrix, keeping a boundary."""
    result = np.zeros_like(matrix)
    result[:boundary_thickness, :] = matrix[:boundary_thickness, :]
    result[-boundary_thickness:, :] = matrix[-boundary_thickness:, :]
    result[:, :boundary_thickness] = matrix[:, :boundary_thickness]
    result[:, -boundary_thickness:] = matrix[:, -boundary_thickness:]
    return result

def create_loader(prefix, base_dir, save_dir, boundary_thickness=2, batch_size=300, num_cells_in_patch=32, shuffle=True, split_train_val=False):
    # Load the data
    with open(os.path.join(base_dir, f'{prefix}_depths.pkl'), 'rb') as f:
        depths_numpy = np.array(pickle.load(f))
    with open(os.path.join(base_dir, f'{prefix}_depths_next.pkl'), 'rb') as f:
        depths_next_numpy = np.array(pickle.load(f))
    with open(os.path.join(base_dir, f'{prefix}_terrains.pkl'), 'rb') as f:
        terrains_numpy = np.array(pickle.load(f))

    # Prepare tensors
    N = terrains_numpy.shape[0]
    terrains_tensor = torch.tensor(np.expand_dims(terrains_numpy, axis=1)).float()
    dataset_numpy = np.zeros((N, 2, num_cells_in_patch, num_cells_in_patch), dtype=np.float32)

    depths_BC_numpy = np.stack([zero_internal(matrix, boundary_thickness) for matrix in depths_next_numpy], axis=0)
    for i in range(N):
        dataset_numpy[i] = np.stack((depths_BC_numpy[i], depths_numpy[i]))

    labels_numpy = np.expand_dims(depths_next_numpy - depths_numpy, axis=1)
    dataset_tensor = torch.tensor(dataset_numpy).float()
    labels_tensor = torch.tensor(labels_numpy).float()

    # Shuffle
    indices = torch.randperm(N)
    dataset_tensor = dataset_tensor[indices]
    labels_tensor = labels_tensor[indices]
    terrains_tensor = terrains_tensor[indices]

    # Split or full loader
    if split_train_val:
        train_size = int(N * 0.7)
        train_dataset = TensorDataset(terrains_tensor[:train_size], dataset_tensor[:train_size], labels_tensor[:train_size])
        val_dataset = TensorDataset(terrains_tensor[train_size:], dataset_tensor[train_size:], labels_tensor[train_size:])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        torch.save(train_loader, os.path.join(save_dir, f'{prefix}_train_loader.pt'))
        torch.save(val_loader, os.path.join(save_dir, f'{prefix}_val_loader.pt'))

        return len(train_loader), len(val_loader)
    else:
        full_dataset = TensorDataset(terrains_tensor, dataset_tensor, labels_tensor)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

        torch.save(full_loader, os.path.join(save_dir, f'{prefix}_loader.pt'))
        return len(full_loader), None

# Paths (relative to project root)
base_dir = './Database/'
save_dir = './Dataloaders/'

# Create save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# List of datasets to process
loader_configs = [
    ('big_train_val', True),
    ('big_test', False),
    ('small_train_val', True),
    ('prj_03_train_val', False),
    ('prj_03_test', False)
]

# Run loader creation
results = {}
for prefix, split in loader_configs:
    results[prefix] = create_loader(prefix, base_dir, save_dir, split_train_val=split)

print("Done. DataLoader sizes:", results)
