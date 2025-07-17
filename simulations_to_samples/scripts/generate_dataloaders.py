# simulations_to_samples/scripts/generate_dataloaders.py

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from config import DATASETS_ROOT, DATALOADERS_ROOT, PATCH_SIZE, BOUNDARY_THICKNESS, BATCH_SIZE


def zero_internal(matrix):
    """Zero out the internal region of a 2D matrix, keeping a boundary."""
    result = np.zeros_like(matrix)
    result[:BOUNDARY_THICKNESS, :] = matrix[:BOUNDARY_THICKNESS, :]
    result[-BOUNDARY_THICKNESS:, :] = matrix[-BOUNDARY_THICKNESS:, :]
    result[:, :BOUNDARY_THICKNESS] = matrix[:, :BOUNDARY_THICKNESS]
    result[:, -BOUNDARY_THICKNESS:] = matrix[:, -BOUNDARY_THICKNESS:]
    return result


def create_loader(prefix, shuffle=True, split_train_val=False, save_test_loader=False):
    # Determine subdirectory
    if prefix.startswith('big_'):
        subdir = 'big_dataset'
    elif prefix.startswith('small_'):
        subdir = 'small_dataset'
    elif prefix.startswith('prj_03'):
        subdir = 'prj_03_dataset'
    else:
        raise ValueError(f"Unknown dataset prefix: {prefix}")
    
    # Load dataset
    
    # Resolve correct filename
    if split_train_val:
        filename = f'{prefix}train_val.pkl'
    elif save_test_loader:
        filename = f'{prefix}test.pkl'
    else:
        raise ValueError("Must set either split_train_val=True or save_test_loader=True")

    filepath = os.path.join(DATASETS_ROOT, subdir, filename)
    with open(filepath, 'rb') as f:
        dataset_dict = pickle.load(f)


    depths_numpy = np.array(dataset_dict['depth'])
    depths_next_numpy = np.array(dataset_dict['depth_next'])
    terrains_numpy = np.array(dataset_dict['terrain'])

    N = terrains_numpy.shape[0]
    terrains_tensor = torch.tensor(np.expand_dims(terrains_numpy, axis=1)).float()
    dataset_numpy = np.zeros((N, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

    depths_BC_numpy = np.stack([zero_internal(matrix) for matrix in depths_next_numpy], axis=0)
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

    os.makedirs(DATALOADERS_ROOT, exist_ok=True)

    # Split into train/val
    if split_train_val:
        train_size = int(N * 0.7)
        train_dataset = TensorDataset(terrains_tensor[:train_size],
                                      dataset_tensor[:train_size],
                                      labels_tensor[:train_size])
        val_dataset = TensorDataset(terrains_tensor[train_size:],
                                    dataset_tensor[train_size:],
                                    labels_tensor[train_size:])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        torch.save(train_loader, os.path.join(DATALOADERS_ROOT, f'{prefix}train_loader.pt'))
        torch.save(val_loader, os.path.join(DATALOADERS_ROOT, f'{prefix}val_loader.pt'))

        return len(train_loader), len(val_loader)

    # Save full test loader
    elif save_test_loader:
        full_dataset = TensorDataset(terrains_tensor, dataset_tensor, labels_tensor)
        full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

        torch.save(full_loader, os.path.join(DATALOADERS_ROOT, f'{prefix}test_loader.pt'))
        return len(full_loader), None

    else:
        train_val_dataset = TensorDataset(terrains_tensor[:],
                                          dataset_tensor[:],
                                          labels_tensor[:])

        train_val_loader = DataLoader(train_val_dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

        torch.save(train_val_loader, os.path.join(DATALOADERS_ROOT, f'{prefix}train_val_loader.pt'))
        return len(train_val_loader), len(train_val_loader)



def plot_samples(loader, prefix, samples_to_plot=10):
    """
    Plot and save sample images from a DataLoader.

    Parameters:
    - loader: DataLoader object
    - prefix: Identifier for the loader (e.g. 'big_ or 'small', or 'prj_03')
    - samples_to_plot: Number of samples to plot/save
    """
    save_dir = os.path.join(DATALOADERS_ROOT, 'figures', f'{prefix}')
    os.makedirs(save_dir, exist_ok=True)

    for idx, (terrain, data, label) in enumerate(loader):
        if idx >= 1:
            break
        for i in range(data.shape[0]):
            if i >= samples_to_plot:
                break

            fig, axs = plt.subplots(1, 4, figsize=(28, 8))

            im0 = axs[0].imshow(terrain[i].squeeze(), cmap='terrain')
            axs[0].set_title('Terrain (m)', fontsize=35)
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04).ax.tick_params(labelsize=30)

            im1 = axs[1].imshow(data[i, 1].squeeze(), cmap='Blues')
            axs[1].set_title('Water Depth at n (m)', fontsize=35)
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04).ax.tick_params(labelsize=30)

            im2 = axs[2].imshow(data[i, 0].squeeze(), cmap='Blues')
            axs[2].set_title('BC at n+1 (m)', fontsize=35)
            fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04).ax.tick_params(labelsize=20)

            im4 = axs[3].imshow(label[i].squeeze(), cmap='Blues')
            axs[3].set_title('Water Depth Difference (m)', fontsize=35)
            fig.colorbar(im4, ax=axs[3], fraction=0.046, pad=0.04).ax.tick_params(labelsize=20)

            plt.tight_layout()
            filepath = os.path.join(save_dir, f'{prefix}_{i}.png')
            plt.savefig(filepath)
            plt.close()

def create_and_save_dataloaders():
    """Run full DataLoader generation pipeline."""

    configs = [
        # (prefix, split, is_test)
        ('small_', True, False),
        ('big_', True, False),
        ('big_', False, True),
        ('prj_03_', False, False),
        ('prj_03_', False, True)
    ]

    for prefix, split, is_test in configs:
        print(f"Now Building {prefix} - split: {split}, test: {is_test}")
        create_loader(prefix, split_train_val=split, save_test_loader=is_test)

        # Visualize only training loaders
        if split:
            loader_path = os.path.join(DATALOADERS_ROOT, f'{prefix}train_loader.pt')
        elif is_test:
            loader_path = os.path.join(DATALOADERS_ROOT, f'{prefix}test_loader.pt')
        else:
            continue  # Skip visualization if unclear

        try:
            loader = torch.load(loader_path)
            plot_samples(loader, prefix=prefix)
        except Exception as e:
            print(f"⚠️ Could not visualize samples for {prefix}: {e}")

