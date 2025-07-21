import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
import os

SCALE = 10000
vmin = -100
vmax = 100


def plot_samples_diff_eval(epoch, which, Architecture_num, trial_num, prediction_diffs, true_diffs, terrains, water_depth, plot_dir, num_samples=10):
    diff = torch.abs(true_diffs - prediction_diffs)
    # Calculate Relative Absolute Error
    diff_sum = torch.sum(diff, dim=(1, 2, 3))
    dummy_diff = torch.abs(true_diffs)
    dummy_diff_sum = torch.sum(dummy_diff, dim=(1, 2, 3))
    RAE_batch = diff_sum / dummy_diff_sum
    RAE_batch_mean = torch.mean(RAE_batch).item()
    RAE_batch_max = torch.max(RAE_batch).item()
    RAE_batch_median = torch.median(RAE_batch).item()


    scaled_diff = diff * SCALE
    scaled_diff = scaled_diff.cpu().numpy()

    random_indices = random.sample(range(prediction_diffs.shape[0]), num_samples)
    prediction_diffs = prediction_diffs.cpu().numpy()
    true_diffs = true_diffs.cpu().numpy()

    for i, sample_index in enumerate(random_indices):
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, height_ratios=[1, 1])
        # First Row
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(prediction_diffs[sample_index].squeeze() * SCALE, cmap='viridis', vmin=-SCALE, vmax=SCALE)
        ax1.set_title('Predicted Depth Diff')
        ax1.axis('off')
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(true_diffs[sample_index].squeeze() * SCALE, cmap='viridis', vmin=-SCALE, vmax=SCALE)
        ax2.set_title('True Depth Diff')
        ax2.axis('off')
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Diff plot
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(scaled_diff[sample_index].squeeze(), cmap='seismic')
        ax3.set_title(f'Diff (True - Pred) - {RAE_batch[sample_index].item():.2f}')
        ax3.axis('off')
        # white to red colors
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # First Histogram
        ax_hist1 = fig.add_subplot(gs[0, 3])
        diff_flat = scaled_diff.flatten()
        diff_flat = diff_flat[diff_flat != 0]
        hist_data, bins, _ = ax_hist1.hist(diff_flat, bins=50, color='blue', edgecolor='black')
        ax_hist1.set_title('Histogram (Diff)')
        ax_hist1.set_xlim(cbar3.vmin, cbar3.vmax)
        ax_hist1.set_xlabel('Value')
        ax_hist1.set_ylabel('Count')

        # Second Row
        # plot terrain
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(terrains[sample_index].numpy().squeeze(), cmap='viridis')
        ax4.set_title('Terrain')
        ax4.axis('off')
        # cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        # ax4.set_xlim(cbar4.vmin, cbar4.vmax)

        # plot water depth
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(water_depth[sample_index].numpy().squeeze(), cmap='viridis')
        ax5.set_title('Water Depth')
        ax5.axis('off')
        # cbar5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        # ax5.set_xlim(cbar5.vmin, cbar5.vmax)


        plt.suptitle(
            f'{Architecture_num}_{trial_num}\n'
            f'epoch_num = {epoch}\n'
            f'{which}',
            size=20, x=0.6)

        # Save each plot in the designated folder
        plt_path = os.path.join(plot_dir, f'{sample_index}.png')
        plt.savefig(plt_path)

        # plt.show()
        plt.close()

    return RAE_batch_mean, RAE_batch_max, RAE_batch_median
