import os
import torch
import matplotlib.pyplot as plt

def plot_all_losses():
    # Set the font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Nimbus Roman']

    # Define local base directory (relative to project root)
    base_dir = 'multi_architecture_training/A_train_all_archs_on_small_set'
    architectures = ['Arch_02', 'Arch_03', 'Arch_04', 'Arch_05', 'Arch_07', 'Arch_08', 'Arch_09']
    save_path = os.path.join(base_dir, 'all_archs_initial_config_losses.png')

    # Load losses
    train_losses, val_losses = {}, {}
    for arch in architectures:
        path = os.path.join(base_dir, 'saved_losses', arch, 'losses.pt')
        if os.path.exists(path):
            data = torch.load(path)
            train_losses[arch] = data.get('train_losses', [])
            val_losses[arch] = data.get('val_losses', [])

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    for arch in architectures:
        ax1.plot(train_losses[arch], label=arch, linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=25)
    ax1.set_ylabel('L1 Loss', fontsize=25)
    ax1.set_title('Training Losses Across Architectures', fontsize=30)
    ax1.tick_params(labelsize=20)
    ax1.grid(True)

    for arch in architectures:
        ax2.plot(val_losses[arch], label=arch, linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=25)
    ax2.set_title('Validation Losses Across Architectures', fontsize=30)
    ax2.tick_params(labelsize=20)
    ax2.grid(True)

    # Combine and deduplicate legends
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    unique = dict(zip(labels + labels2, handles + handles2))
    fig.legend(unique.values(), unique.keys(), loc='upper center', fontsize=25,
               ncol=max(1, len(unique) // 2), bbox_to_anchor=(0.5, -0.05))

    plt.subplots_adjust(bottom=0.09)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
