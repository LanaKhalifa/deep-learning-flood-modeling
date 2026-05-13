import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
import matplotlib.font_manager as font_manager
from multi_architecture_training.models.non_downsampling_convolutions_attention import NonDownsamplingConvolutionsWithAttention
from multi_architecture_training.models.terrain_downsampler_alternating import TerrainDownsampleAlternating
from config.paths_config import DATALOADERS_DIR

# Set the font to Nimbus Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman']


def plot_ten_predictions(idx, prediction_diffs, true_diffs, terrains, water_depth, plot_dir):
    num_samples = 10
    diff = torch.abs(true_diffs - prediction_diffs)
    non_abs_diff = true_diffs - prediction_diffs
    # Calculate Relative Absolute Error
    diff_sum = torch.sum(diff, dim=(1, 2, 3))
    dummy_diff = torch.abs(true_diffs)
    dummy_diff_sum = torch.sum(dummy_diff, dim=(1, 2, 3))
    RAE_batch = diff_sum / dummy_diff_sum
    RAE_batch_mean = torch.mean(RAE_batch).item()
    RAE_batch_max = torch.max(RAE_batch).item()
    RAE_batch_median = torch.median(RAE_batch).item()
    diff = diff.cpu().numpy()
    non_abs_diff = non_abs_diff.cpu().numpy()
    indices = np.arange(num_samples)
    prediction_diffs = prediction_diffs.cpu().numpy()
    true_diffs = true_diffs.cpu().numpy()
    terrains = terrains.cpu().numpy()
    water_depth = water_depth.cpu().numpy()
    fig = plt.figure(figsize=(65, 15*num_samples))
    gs = GridSpec(num_samples, 5, width_ratios=[1, 10, 10, 10, 10])
    column_titles = ['RAE\n[-]', 'Prediction\n(m)', 'Ground Truth\n(m)', 'Absolute Error\n(m)', 'Error Histogram\n(m)']
    for col, title in enumerate(column_titles):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(title, fontsize=150, pad=200)
        ax.axis('off')
    for row, sample_index in enumerate(indices):
        ax_text = fig.add_subplot(gs[row, 0])
        ax_text.text(0.5, 0.5, f'{RAE_batch[sample_index].item():.2f}', fontsize=100, ha='center', va='center')
        ax_text.axis('off')
        max_for_colorbar_1 = np.max(np.abs(prediction_diffs[sample_index]))
        max_for_colorbar_2 = np.max(np.abs(true_diffs[sample_index]))    
        max_for_colorbar = np.max([max_for_colorbar_1,max_for_colorbar_2])
        ax5 = fig.add_subplot(gs[row, 1])
        im5 = ax5.imshow(prediction_diffs[sample_index].squeeze(), cmap='BrBG', vmin=-max_for_colorbar, vmax=max_for_colorbar)
        ax5.axis('off')
        cbar5 = fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=50, length=20, width = 5)
        ax6 = fig.add_subplot(gs[row, 2])
        im6 = ax6.imshow(true_diffs[sample_index].squeeze(), cmap='BrBG', vmin=-max_for_colorbar, vmax=max_for_colorbar)
        ax6.axis('off')
        cbar6 = fig.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        cbar6.ax.tick_params(labelsize=50, length=20, width = 5)
        ax7 = fig.add_subplot(gs[row, 3])
        m = np.max(diff[sample_index])
        im7 = ax7.imshow(diff[sample_index].squeeze(), cmap='seismic', vmin=0, vmax=m)
        ax7.axis('off')
        cbar7 = fig.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
        cbar7.ax.tick_params(labelsize=50, length=20, width = 5)
        ax_hist1 = fig.add_subplot(gs[row, 4])
        diff_flat = non_abs_diff[sample_index].flatten()
        hist_data, bins, _ = ax_hist1.hist(diff_flat, bins=50, color='black', edgecolor='black')
        ax_hist1.set_xlim(-m, m)
        ax_hist1.yaxis.set_label_position("right")
        ax_hist1.yaxis.tick_right()
        ax_hist1.tick_params(axis='both', which='major', labelsize=50, length=20, width=5)
    plt.tight_layout()
    plt_path = os.path.join(plot_dir, f'ten_predictions_{idx}.png')
    plt.savefig(plt_path)
    plt.show()
    plt.close()
    return RAE_batch_mean, RAE_batch_max, RAE_batch_median


def plot_entire_batch(plot_dir='evaluate_and_visualize_best_model/visual_predictions'):
    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    downsampler = TerrainDownsampleAlternating(c_start=1, c1=20, c2=40, c_end=1).to(device)
    model = NonDownsamplingConvolutionsWithAttention(
        downsampler=downsampler,
        arch_input_c=3,
        arch_num_layers=12,
        arch_num_c=32,
        arch_act="leakyrelu",
        arch_last_act="leakyrelu",
        arch_num_attentions=2
    ).to(device)
    model_path = 'multi_architecture_training/C_train_best_four_on_big_set/saved_trained_models/Arch_04/model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dataloader_path = os.path.join(DATALOADERS_DIR, 'big_val_loader.pt')
    dataloader = torch.load(dataloader_path)
    terrains, data, labels = next(iter(dataloader))
    terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)
    with torch.no_grad():
        predictions = model(terrains, data)
    num_samples = terrains.shape[0]
    num_per_fig = 10
    num_figs = num_samples // num_per_fig
    for idx in range(num_figs):
        start = idx * num_per_fig
        end = start + num_per_fig
        plot_ten_predictions(
            idx,
            prediction_diffs=predictions[start:end],
            true_diffs=labels[start:end],
            terrains=terrains[start:end],
            water_depth=data[start:end, 1],
            plot_dir=plot_dir
        ) 