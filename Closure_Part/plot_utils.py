
"""
plot_utils.py

Plotting utilities for closure results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from config import FONT_SIZE, TICK_INTERVAL

def plot_convergence(next_old_diffs, true_pred_diffs, dummy_perf, save_path):
    """
    Plots MAE curves for closure loop convergence.

    Args:
        next_old_diffs (list): MAE of consecutive predictions
        true_pred_diffs (list): MAE to ground truth
        dummy_perf (float): Dummy MAE baseline
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(60, 20))

    ax1.plot(next_old_diffs, label='Predicted_Current - Predicted_Old')
    ax1.set_title('Predicted_Current - Predicted_Old', fontsize=FONT_SIZE)
    ax1.set_xlabel('Iteration', fontsize=FONT_SIZE)
    ax1.set_ylabel('MAE', fontsize=FONT_SIZE)
    ax1.legend(fontsize=FONT_SIZE)

    ax2.plot(true_pred_diffs, label='Ground_Truth - Predicted_Current')
    ax2.axhline(y=dummy_perf, color='r', linestyle='--', label='Dummy')
    ax2.set_title('Ground_Truth - Predicted_Current', fontsize=FONT_SIZE)
    ax2.set_xlabel('Iteration', fontsize=FONT_SIZE)
    ax2.set_ylabel('MAE', fontsize=FONT_SIZE)
    ax2.legend(fontsize=FONT_SIZE)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_comparison_matrices(depth_next, depth_next_dummy, true_next,
                             l1_val, rae_val, save_path):
    """
    Plots 9-matrix comparison layout.

    Args:
        depth_next: Final predicted matrix
        depth_next_dummy: Dummy matrix
        true_next: Ground truth matrix
        l1_val: Final L1
        rae_val: Final RAE
        save_path: Save path
    """
    diff_matrix = np.abs(true_next - depth_next)
    diff_dummy = np.abs(true_next - depth_next_dummy)
    pred_vs_dummy = np.abs(depth_next - depth_next_dummy)

    vmax = max(depth_next.max(), true_next.max(), depth_next_dummy.max())
    vmin = min(depth_next.min(), true_next.min(), depth_next_dummy.min())
    diff_max = max(diff_matrix.max(), diff_dummy.max())

    fig, axes = plt.subplots(3, 3, figsize=(80, 55))
    cmap_diff = mcolors.LinearSegmentedColormap.from_list("white_red", [(1, 1, 1), (1, 0, 0)])
    cmap_green = mcolors.LinearSegmentedColormap.from_list("greens", [(0.9, 1, 0.9), (0, 0.5, 0)])

    def plot_one(ax, data, title, cmap, vmin=None, vmax=None):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.33, pad=0.04)
        cbar.ax.tick_params(labelsize=60)
        ax.set_title(title, fontsize=100)
        ax.set_xticks(np.arange(0, data.shape[1], TICK_INTERVAL))
        ax.set_yticks(np.arange(0, data.shape[0], TICK_INTERVAL))
        ax.tick_params(axis='x', labelsize=60)
        ax.tick_params(axis='y', labelsize=60)

    plot_one(axes[0, 0], depth_next, "Predicted Depth Next (m)", "Blues", vmin, vmax)
    plot_one(axes[0, 1], true_next, "Ground Truth Next (m)", "Blues", vmin, vmax)
    plot_one(axes[0, 2], diff_matrix, f'L1={l1_val:.2f}, RAE={rae_val:.2f}\n|Truth - Predicted| (m)', cmap_diff, 0, diff_max)

    plot_one(axes[1, 0], depth_next_dummy, "Dummy Depth Next (m)", "Blues", vmin, vmax)
    plot_one(axes[1, 1], true_next, "Ground Truth Next (m)", "Blues", vmin, vmax)
    plot_one(axes[1, 2], diff_dummy, "|Truth - Dummy| (m)", cmap_diff, 0, diff_max)

    plot_one(axes[2, 0], depth_next_dummy, "Dummy Depth Next (m)", "Blues", vmin, vmax)
    plot_one(axes[2, 1], depth_next, "Predicted Depth Next (m)", "Blues", vmin, vmax)
    plot_one(axes[2, 2], pred_vs_dummy, "|Predicted - Dummy| (m)", cmap_green, 0, diff_max)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
