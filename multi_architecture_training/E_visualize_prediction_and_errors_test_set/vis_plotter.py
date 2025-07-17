import os
import torch
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from config import ROOT_DATALOADERS

def visualize_predictions_on_test_sets(model, device, arch_name, save_root, num_samples=10):
    test_loaders = {
        'prj_03_test': 'prj_03_test_loader.pt',
        'big_test': 'big_test_loader.pt'
    }

    for set_name, loader_file in test_loaders.items():
        loader_path = os.path.join(ROOT_DATALOADERS, loader_file)
        loader = torch.load(loader_path)

        base_dir = os.path.join(save_root, arch_name, set_name)
        os.makedirs(base_dir, exist_ok=True)

        # Use only first batch for visualization
        for terrains, data, labels in loader:
            terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)
            with torch.no_grad():
                preds = model(terrains, data)
            plot_prediction_error_maps(preds, labels, base_dir, num_samples=num_samples, epoch=0, which=set_name)
            break

def plot_prediction_error_maps(preds, labels, base_dir, num_samples=10, epoch=0, which=""):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    indices = random.sample(range(preds.shape[0]), num_samples)

    for idx in indices:
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(1, 4)

        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(preds[idx].squeeze(), cmap='viridis')
        ax1.set_title("Predicted Depth Diff")
        ax1.axis("off")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(labels[idx].squeeze(), cmap='viridis')
        ax2.set_title("True Depth Diff")
        ax2.axis("off")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        diff = labels[idx] - preds[idx]
        im3 = ax3.imshow(diff.squeeze(), cmap='viridis')
        ax3.set_title("Diff (True - Pred)")
        ax3.axis("off")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4 = fig.add_subplot(gs[0, 3])
        diff_flat = diff.flatten()
        diff_flat = diff_flat[diff_flat != 0]
        ax4.hist(diff_flat, bins=50, color='blue', edgecolor='black')
        ax4.set_title("Histogram (Diff)")
        ax4.set_xlabel("Value")
        ax4.set_ylabel("Count")

        plt.suptitle(f'epoch = {epoch}\n{which}', size=20, x=0.6)

        save_path = os.path.join(base_dir, f"epoch_{epoch}_sample_{idx}.png")
        plt.savefig(save_path)
        plt.close()
