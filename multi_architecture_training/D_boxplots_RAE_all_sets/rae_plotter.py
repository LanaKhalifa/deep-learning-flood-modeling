# multi_architecture_training/D_boxplots_RAE_all_sets/rae_plotter.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from config import ROOT_DATALOADERS  # Import centralized loader paths
import os

def calculate_rae_all_quartiles(netG, device, save_path):
    netG.eval()

    all_rae_arrays = []
    box_labels = []

    for label, loader_path in ROOT_DATALOADERS.items():
        loader = torch.load(loader_path)
        raes_all_samples = []

        with torch.no_grad():
            for terrains, data, labels in loader:
                terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)

                preds = netG(terrains, data)
                diff_sum = torch.abs(labels - preds).sum(dim=(1, 2, 3))
                dummy_sum = torch.abs(labels).sum(dim=(1, 2, 3))

                batch_rae = diff_sum / dummy_sum
                raes_all_samples.extend(batch_rae.cpu().numpy())

        rae_array = np.array(raes_all_samples)
        rae_array = rae_array[np.isfinite(rae_array)]

        # Remove outliers using IQR
        Q1 = np.percentile(rae_array, 25)
        Q3 = np.percentile(rae_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered = rae_array[(rae_array >= lower_bound) & (rae_array <= upper_bound)]

        all_rae_arrays.append(filtered)
        box_labels.append(label)

    # Plot
    fig, ax = plt.subplots(figsize=(1 + len(all_rae_arrays), 10))

    ax.boxplot(all_rae_arrays,
               vert=True,
               patch_artist=True,
               showmeans=False,
               showfliers=False,
               medianprops=dict(color='blue', linewidth=2),
               boxprops=dict(color='black', linewidth=2, facecolor='lightgrey'),
               whiskerprops=dict(color='black', linewidth=2),
               capprops=dict(color='black', linewidth=2),
               widths=0.5)

    ax.set_xticks(np.arange(1, len(box_labels) + 1))
    ax.set_xticklabels(box_labels, fontsize=18, rotation=20)
    ax.tick_params(axis='y', labelsize=20, labelright=True, labelleft=False)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.25)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
