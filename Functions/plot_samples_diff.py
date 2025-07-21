import random
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
import os

def plot_samples_diff(epoch, which, prediction_diffs, true_diffs, base_dir, num_samples = 10):
    random_indices = random.sample(range(prediction_diffs.shape[0]), num_samples)
    prediction_diffs = prediction_diffs.detach().cpu().numpy()
    true_diffs = true_diffs.detach().cpu().numpy()
    
    for i, sample_index in enumerate(random_indices):
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(1, 4, height_ratios=[1])

        # First Row
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(prediction_diffs[sample_index].squeeze(), cmap='viridis')
        ax1.set_title('Predicted Depth Diff')
        ax1.axis('off')
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(true_diffs[sample_index].squeeze(), cmap='viridis')
        ax2.set_title('True Depth Diff')
        ax2.axis('off')
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        diff = true_diffs[sample_index] - prediction_diffs[sample_index]
        im3 = ax3.imshow(diff.squeeze(), cmap='viridis')
        ax3.set_title('Diff (True - Pred)')
        ax3.axis('off')
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # First Histogram
        ax_hist1 = fig.add_subplot(gs[0, 3])
        diff_flat = diff.flatten()
        diff_flat = diff_flat[diff_flat != 0]
        hist_data, bins, _ = ax_hist1.hist(diff_flat, bins=50, color='blue', edgecolor='black')
        ax_hist1.set_title('Histogram (Diff)')
        ax_hist1.set_xlim(cbar3.vmin, cbar3.vmax)
        ax_hist1.set_xlabel('Value')
        ax_hist1.set_ylabel('Count')
        
        plt.suptitle(
            f'epoch_num = {epoch}\n'
            f'{which}',
            size=20, x=0.6)

        # Save each plot in the designated folder
        plt_path = os.path.join(base_dir, f'epoch_{epoch}_sample_{sample_index}.png')
        plt.savefig(plt_path)
        
        plt.show()
        plt.close()