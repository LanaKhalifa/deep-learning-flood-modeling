# multi_architecture_training/D_boxplots_RAE_all_sets/rae_plotter.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import ROOT_DATALOADERS

def calculate_rae_boxplot_all_sets(netG, device, save_path):
    netG.eval()

    all_rae_arrays = []
    box_labels = []

    for fname in sorted(os.listdir(ROOT_DATALOADERS)):
        if not fname.endswith(".pt"):
            continue

        label = fname.replace("_loader.pt", "").replace("_", " ").title()  # e.g. "prj_01_test_loader.pt" → "Prj 01 Test"
        full_path = os.path.join(ROOT_DATALOADERS, fname)

        loader = torch.load(full_path)
        raes_all_samples = []

        with torch.no_grad():
            for terrains, data, labels in loader:
                terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)
                preds = netG(terrains, data)
                diff_sum = torch.abs(labels - preds).sum(dim=(1, 2, 3))
                dummy_sum = torch.abs(labels).sum(dim=(1, 2, 3))
                rae_batch = diff_sum / dummy_sum
                raes_all_samples.extend(rae_batch.cpu().numpy())

        rae_array = np.array(raes_all_samples)
        rae_array = rae_array[np.isfinite(rae_array)]

        # Filter out outliers using IQR
        Q1 = np.percentile(rae_array, 25)
        Q3 = np.percentile(rae_array, 75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        filtered = rae_array[(rae_array >= lower) & (rae_array <= upper)]

        all_rae_arrays.append(filtered)
        box_labels.append(label)

    # Create boxplot
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
