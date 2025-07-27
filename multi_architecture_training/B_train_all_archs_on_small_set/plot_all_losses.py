import torch
import matplotlib.pyplot as plt
from pathlib import Path
from config.paths_config import STAGE_B_DIR, get_losses_path, TRAINING_UTILS_DIR

def plot_all_losses():
    """
    Plot training and validation losses for all architectures in Stage B.
    Uses the new config system for paths.
    """
    # Set font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Nimbus Roman']

    # Define paths using config
    base_dir = STAGE_B_DIR
    architectures = ['Arch_02', 'Arch_03', 'Arch_04', 'Arch_05', 'Arch_07', 'Arch_08', 'Arch_09']
    
    # Architecture name mapping
    arch_names = {
        'Arch_02': 'Non-Downsampling Convolutions',
        'Arch_03': 'Simplified UNet',
        'Arch_04': 'Non-Downsampling Convolutions + Attention',
        'Arch_05': 'Classic UNet',
        'Arch_07': 'Encoder-Decoder + Attention',
        'Arch_08': 'UNet + ResNet Modified',
        'Arch_09': 'Encoder-Decoder + Large Convolutions'
    }
    
    # Define consistent colors for architectures (same as C)
    colors = {
        'Arch_03': '#1f77b4',  # blue
        'Arch_04': '#ff7f0e',  # orange
        'Arch_05': '#2ca02c',  # green
        'Arch_07': '#d62728',  # red
        'Arch_02': '#9467bd',  # purple
        'Arch_08': '#8c564b',  # brown
        'Arch_09': '#e377c2',  # pink
    }
    
    save_path = base_dir / 'figures' / 'learning_curves.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize containers
    train_losses, val_losses = {}, {}
    dummy_val_loss = torch.load(TRAINING_UTILS_DIR / 'dummy_small_val_loss.pt')

    # Load losses using config paths
    for arch in architectures:
        losses_path = get_losses_path('B', arch)
        data = torch.load(losses_path)
        train_losses[arch] = data.get('train_losses', [])
        val_losses[arch] = data.get('val_losses', [])

    # Calculate common axis limits
    all_losses = [dummy_val_loss]
    max_epochs = 0
    for arch in architectures:
        all_losses.extend(train_losses[arch])
        all_losses.extend(val_losses[arch])
        max_epochs = max(max_epochs, len(train_losses[arch]), len(val_losses[arch]))
    
    y_min = min(all_losses) * 0.95  # Add 5% margin
    y_max = max(all_losses) * 1.05  # Add 5% margin
    x_max = max_epochs - 1

    # Create plot with more compact dimensions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Training All Architectures on Small Dataset', fontsize=35, y=1.02)

    # --- TRAIN LOSSES ---
    for arch in architectures:
        ax1.plot(train_losses[arch], label=arch_names[arch], linewidth=2, color=colors[arch])

    ax1.set_xlabel('Epoch', fontsize=25)
    ax1.set_ylabel('L1 Loss', fontsize=25)
    ax1.set_title('Train Loss', fontsize=30)
    ax1.tick_params(labelsize=20)
    ax1.grid(True, alpha=0.3)
    # Set common axis limits
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(y_min, 0.5)

    # --- VAL LOSSES ---
    for arch in architectures:
        ax2.plot(val_losses[arch], label=arch_names[arch], linewidth=2, color=colors[arch])
    ax2.axhline(y=dummy_val_loss, color='green', linestyle='--',
                label='Mean Dummy Val Loss', linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=25)
    ax2.set_title('Val Loss', fontsize=30)
    ax2.tick_params(labelsize=20)
    ax2.grid(True, alpha=0.3)
    # Set common axis limits
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(y_min, 0.5)

    # --- LEGEND ---
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    unique = dict(zip(labels + labels2, handles + handles2))
    
    # Adjust legend size to accommodate 8 entries (7 architectures + 1 dummy line)
    fig.legend(unique.values(), unique.keys(), loc='upper center', fontsize=22,
               ncol=3, bbox_to_anchor=(0.5, 0.02))

    # Use more compact subplot adjustment
    plt.subplots_adjust(bottom=0.15, top=0.88, left=0.05, right=0.95, wspace=0.3)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
