# multi_architecture_training/D_boxplots_RAE_all_sets/rae_utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt

def calculate_rae_quartiles(model, loader, device, save_path=None, box_width=0.4, figsize=(2, 10), label=None):
    model.eval()
    raes = []

    with torch.no_grad():
        for terrains, data, labels in loader:
            terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)
            preds = model(terrains, data)

            abs_diff = torch.abs(labels - preds).sum(dim=(1, 2, 3))
            abs_dummy = torch.abs(labels).sum(dim=(1, 2, 3))
            batch_rae = abs_diff / abs_dummy

            raes.extend(batch_rae.cpu().numpy())

    rae_array = np.array(raes)
    rae_array = rae_array[np.isfinite(rae_array)]

    Q1, Q2, Q3 = np.percentile(rae_array, [25, 50, 75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    filtered = rae_array[(rae_array >= lower) & (rae_array <= upper)]
    min_val, max_val = filtered.min(), filtered.max()

    if save_path:
        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot([rae_array], vert=True, patch_artist=True, showmeans=False, showfliers=False,
                   medianprops=dict(color='blue', linewidth=2),
                   boxprops=dict(color='black', linewidth=2, facecolor='lightgrey'),
                   whiskerprops=dict(color='black', linewidth=2),
                   capprops=dict(color='black', linewidth=2),
                   widths=box_width)

        if label:
            ax.set_xticks([1])
            ax.set_xticklabels([label], fontsize=18)
        ax.tick_params(axis='y', labelsize=18, labelright=True, labelleft=False)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    return Q1, Q2, Q3, min_val, max_val
