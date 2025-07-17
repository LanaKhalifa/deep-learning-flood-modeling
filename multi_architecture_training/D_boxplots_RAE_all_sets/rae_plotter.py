import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import ROOT_DATALOADERS

def calculate_rae_boxplot_all_sets(model, device, save_path):
    """
    Plots a single figure with RAE boxplots over multiple datasets (loaders).
    Saves the figure at save_path.
    """
    loader_names = [
        'small_val_loader.pt',
        'small_train_loader.pt',
        'small_test_loader.pt',
        'big_val_loader.pt',
        'big_train_loader.pt',
        'big_test_loader.pt',
        'prj_03_test_loader.pt',
        'prj_03_train_val_loader.pt',
    ]
    
    loader_labels = [
        'Small Val',
        'Small Train',
        'Small Test',
        'Big Val',
        'Big Train',
        'Big Test',
        'Prj03 Test',
        'Prj03 Train+Val'
    ]

    all_rae_data = []

    model.eval()

    for loader_file in loader_names:
        loader_path = os.path.join(ROOT_DATALOADERS, loader_file)
        loader = torch.load(loader_path)
        rae_values = []

        with torch.no_grad():
            for terrains, data, labels in loader:
                terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)

                preds = model(terrains, data)

                abs_diff = torch.abs(labels - preds)
                abs_diff_sum = torch.sum(abs_diff, dim=(1, 2, 3))

                abs_baseline = torch.abs(labels)
                abs_baseline_sum = torch.sum(abs_baseline, dim=(1, 2, 3))

                rae = abs_diff_sum / abs_baseline_sum
                rae_values.extend(rae.cpu().numpy())

        rae_array = np.array(rae_values)
        rae_array = rae_array[np.isfinite(rae_array)]  # remove NaN/inf
        all_rae_data.append(rae_array)

    # -----------------------
    # Plotting
    # -----------------------
    fig, ax = plt.subplots(figsize=(8, 10))

    ax.boxplot(all_rae_data, vert=True, patch_artist=True, showmeans=False, showfliers=False,
               medianprops=dict(color='blue', linewidth=2),
               boxprops=dict(color='black', linewidth=2, facecolor='lightgrey'),
               whiskerprops=dict(color='black', linewidth=2),
               capprops=dict(color='black', linewidth=2),
               widths=0.5)

    ax.set_xticks(range(1, len(loader_labels) + 1))
    ax.set_xticklabels(loader_labels, fontsize=14)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.title("Relative Absolute Error (RAE) Across Sets", fontsize=18)
    plt.ylabel("RAE", fontsize=16)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
