# multi_architecture_training/B_tune_one_arch_on_small_set/plot_losses.py

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from config.paths_config import STAGE_B_DIR, get_losses_path, TRAINING_UTILS_DIR


def plot_losses():
    """
    Plot training and validation losses for Arch_04 in Stage B.
    Uses the new config system for paths.
    """
    # Set font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Nimbus Roman']

    # Define paths using config
    base_dir = STAGE_B_DIR
    arch_name = 'Arch_04'
    
    save_path = base_dir / 'tuned_arch_04_loss_plot.png'

    # Load losses using config paths
    losses_path = get_losses_path('B', arch_name)

    data = torch.load(losses_path)
    train_losses = data.get('train_losses', [])
    val_losses = data.get('val_losses', [])
    
    # Load dummy validation loss
    dummy_val_loss = torch.load(TRAINING_UTILS_DIR / 'dummy_small_val_loss.pt')

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Training Tuned Arch_04 on Small Dataset', fontsize=35)

    # --- TRAIN LOSSES ---
    ax1.plot(train_losses, label='Train Loss', linewidth=2, color='blue')
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('L1 Loss', fontsize=20)
    ax1.set_title('Train Loss - Arch_04 (Tuned)', fontsize=24)
    ax1.tick_params(labelsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=16)

    # --- VAL LOSSES ---
    ax2.plot(val_losses, label='Validation Loss', linewidth=2, color='red')
    ax2.axhline(y=dummy_val_loss, color='green', linestyle='--',
                label='Mean Dummy Val Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=20)
    ax2.set_ylabel('L1 Loss', fontsize=20)
    ax2.set_title('Validation Loss - Arch_04 (Tuned)', fontsize=24)
    ax2.tick_params(labelsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    