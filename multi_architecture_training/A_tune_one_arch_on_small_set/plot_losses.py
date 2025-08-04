# multi_architecture_training/A_tune_one_arch_on_small_set/plot_losses.py

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from config.paths_config import STAGE_A_DIR, get_losses_path, TRAINING_UTILS_DIR


def plot_losses():
    """
    Plot training and validation losses for Arch_04 in Stage A.
    Uses the new config system for paths.
    """
    # Set font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Nimbus Roman']

    # Define paths using config
    base_dir = STAGE_A_DIR
    arch_name = 'Arch_04'
    
    save_path = base_dir / 'figures' / 'tuned_arch_04_loss_plot.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Load losses using config paths
    losses_path = get_losses_path('A', arch_name)

    data = torch.load(losses_path)
    train_losses = data.get('train_losses', [])
    val_losses = data.get('val_losses', [])
    
    # Load dummy validation loss
    dummy_val_loss = torch.load(TRAINING_UTILS_DIR / 'dummy_small_val_loss.pt')

    # Calculate common axis limits
    all_losses = train_losses + val_losses + [dummy_val_loss]
    y_min = min(all_losses) * 0.95  # Add 5% margin
    y_max = max(all_losses) * 1.05  # Add 5% margin

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
    # Set common axis limits
    ax1.set_ylim(y_min, y_max)

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
    # Set common axis limits
    ax2.set_xlim(0, 300)
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    