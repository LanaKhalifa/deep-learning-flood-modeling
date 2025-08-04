# multi_architecture_training/C_train_best_four_on_big_set/plot_all_losses.py

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from config.paths_config import STAGE_C_DIR, get_losses_path, TRAINING_UTILS_DIR


def plot_all_losses():
    """
    Plot training and validation losses for the best four architectures in Stage C.
    Uses the new config system for paths.
    """
    # Set font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Nimbus Roman']

    # Define paths using config
    base_dir = STAGE_C_DIR
    architectures = ['Arch_02', 'Arch_04', 'Arch_05']
    
    # Architecture name mapping
    arch_names = {'Arch_02': 'Non-Downsampling Convolutions',
                  'Arch_04': 'Non-Downsampling Convolutions + Attention',
                  'Arch_05': 'Classic UNet'}
    
    # Define consistent colors for architectures (same as A)
    colors = {
        'Arch_02': '#8A2BE2',  # purple
        'Arch_04': '#ff7f0e',  # orange
        'Arch_05': '#2ca02c',  # green
    }
    
    save_path = base_dir / 'learning_curves.png'

    # Initialize containers
    train_losses, val_losses = {}, {}
    dummy_val_loss = torch.load(TRAINING_UTILS_DIR / 'dummy_big_val_loss.pt')

    # Load losses using config paths
    for arch in architectures:
        losses_path = get_losses_path('C', arch)
        data = torch.load(losses_path)
        train_losses[arch] = data.get('train_losses', [])
        val_losses[arch] = data.get('val_losses', [])

    # Create plot with more compact dimensions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Training Best Four Architectures on Big Dataset', fontsize=35, y=1.02)

    # --- TRAIN LOSSES ---
    for arch in architectures:
        ax1.plot(train_losses[arch], label=arch_names[arch], linewidth=2, color=colors[arch])

    ax1.set_xlabel('Epoch', fontsize=25)
    ax1.set_ylabel('L1 Loss', fontsize=25)
    ax1.set_title('Train Loss', fontsize=30)
    ax1.tick_params(labelsize=20)
    ax1.grid(True, alpha=0.3)
    # Limit y-axis to show meaningful range (exclude extreme outliers)
    ax1.set_ylim(0, 0.51)

    # --- VAL LOSSES ---
    for arch in architectures:
        ax2.plot(val_losses[arch], label=arch_names[arch], linewidth=2, color=colors[arch])
        ax2.axhline(y=dummy_val_loss, color='green', linestyle='--',
                    label='Mean Dummy Val Loss', linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=25)
    ax2.set_title('Val Loss', fontsize=30)
    ax2.tick_params(labelsize=20)
    ax2.grid(True, alpha=0.3)
    # Limit y-axis to show meaningful range (exclude loss explosion)
    ax2.set_ylim(0, 0.51)

    # --- LEGEND ---
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    unique = dict(zip(labels + labels2, handles + handles2))
    
    # Adjust legend size to accommodate 5 entries (4 architectures + 1 dummy line)
    fig.legend(unique.values(), unique.keys(), loc='upper center', fontsize=22,
               ncol=3, bbox_to_anchor=(0.5, 0.02))

    # Use more compact subplot adjustment
    plt.subplots_adjust(bottom=0.15, top=0.88, left=0.05, right=0.95, wspace=0.3)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
