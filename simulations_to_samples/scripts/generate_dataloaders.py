# simulations_to_samples/scripts/generate_dataloaders.py

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from config.data_config import PATCH_SIZE, BOUNDARY_THICKNESS
from config.training_config import BATCH_SIZE
from config.paths_config import DATASETS_DIR, DATALOADERS_DIR, PATCHES_DIR

# Set default tensor type to double precision for all PyTorch operations
torch.set_default_dtype(torch.float64)

def zero_internal(matrix):
    """Zero out the internal region of a 2D matrix, keeping a boundary."""
    result = np.zeros_like(matrix)
    result[:BOUNDARY_THICKNESS, :] = matrix[:BOUNDARY_THICKNESS, :]
    result[-BOUNDARY_THICKNESS:, :] = matrix[-BOUNDARY_THICKNESS:, :]
    result[:, :BOUNDARY_THICKNESS] = matrix[:, :BOUNDARY_THICKNESS]
    result[:, -BOUNDARY_THICKNESS:] = matrix[:, -BOUNDARY_THICKNESS:]
    return result

def create_loader(set_name: str, test: bool, train_val_split: bool):
    if test:
        dataset_path = DATASETS_DIR / f'{set_name}_dataset' / f'{set_name}_test.pkl'
    else:
        dataset_path = DATASETS_DIR / f'{set_name}_dataset' / f'{set_name}_train_val.pkl'
    
    with open(dataset_path, 'rb') as f:
        dataset_dict = pickle.load(f)

    # load depth, depth_next, terrain
    depths_numpy = np.array(dataset_dict['depth'])
    depths_next_numpy = np.array(dataset_dict['depth_next'])
    terrains_numpy = np.array(dataset_dict['terrain'])
    # create dataset_tensor and labels_tensor
    N = terrains_numpy.shape[0]
    terrains_tensor = torch.tensor(np.expand_dims(terrains_numpy, axis=1)).double()
    dataset_numpy = np.zeros((N, 2, PATCH_SIZE, PATCH_SIZE), dtype=np.float64)
    depths_BC_numpy = np.stack([zero_internal(matrix) for matrix in depths_next_numpy], axis=0)
    for i in range(N):
        dataset_numpy[i] = np.stack((depths_BC_numpy[i], depths_numpy[i]))
    labels_numpy = np.expand_dims(depths_next_numpy - depths_numpy, axis=1)
    dataset_tensor = torch.tensor(dataset_numpy).double()
    labels_tensor = torch.tensor(labels_numpy).double()
    # Shuffle
    indices = torch.randperm(N)
    dataset_tensor = dataset_tensor[indices]
    labels_tensor = labels_tensor[indices]
    terrains_tensor = terrains_tensor[indices]

    # Split into train/val or test and save dataloader
    if test:
        full_dataset = TensorDataset(terrains_tensor, dataset_tensor, labels_tensor)
        full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
        dataloader_name = f'{set_name}_test_loader'
        torch.save(full_loader, DATALOADERS_DIR / f'{dataloader_name}.pt')
        print(f"Saved {dataloader_name}")
        plot_samples(full_loader, dataloader_name)

    elif train_val_split:
        train_size = int(N * 0.7)
        train_dataset = TensorDataset(terrains_tensor[:train_size],
                                    dataset_tensor[:train_size],
                                    labels_tensor[:train_size])
        val_dataset = TensorDataset(terrains_tensor[train_size:],
                                    dataset_tensor[train_size:],
                                    labels_tensor[train_size:])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        dataloader_name = f'{set_name}_train_loader'
        torch.save(train_loader, DATALOADERS_DIR / f'{dataloader_name}.pt')
        print(f"Saved {dataloader_name}")
        plot_samples(train_loader, dataloader_name)

        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        dataloader_name = f'{set_name}_val_loader'
        torch.save(val_loader, DATALOADERS_DIR / f'{dataloader_name}.pt')
        print(f"Saved {dataloader_name}")
        plot_samples(val_loader, dataloader_name)

    else:
        train_val_dataset = TensorDataset(terrains_tensor[:],
                                        dataset_tensor[:],
                                        labels_tensor[:])

        train_val_loader = DataLoader(train_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
        dataloader_name = f'{set_name}_train_val_loader'
        torch.save(train_val_loader, DATALOADERS_DIR / f'{dataloader_name}.pt')
        print(f"Saved {dataloader_name}")
        plot_samples(train_val_loader, dataloader_name)

def plot_samples(loader, dataloader_name, samples_to_plot=10):
    """
    Plot and save sample images from a DataLoader.

    Parameters:
    - loader: DataLoader object
    - prefix: Identifier for the loader (e.g. 'big_ or 'small', or 'prj_03')
    - samples_to_plot: Number of samples to plot/save
    """
    save_dir = DATALOADERS_DIR / 'figures' / f'{dataloader_name}'
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
            filepath = os.path.join(save_dir, f'sample_{i}.png')
            plt.savefig(filepath)
            plt.close()

def create_and_save_dataloaders():
    os.makedirs(DATALOADERS_DIR, exist_ok=True)
    """Run full DataLoader generation pipeline."""
    set_names = ['big', 'small', 'prj_03']
    configs = [
        # (dataset_name, test, train_val_split)
        ('big', True, False),
        ('big', False, True),
        ('small', False, True),
        ('prj_03', True, False),
        ('prj_03', False, False)
    ]
    for set_name, test, train_val_split in configs:
        create_loader(set_name, test, train_val_split)
