import os
import torch
import matplotlib.pyplot as plt

def plot_all_losses():
    # Set font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Nimbus Roman']

    # Define paths
    base_dir = 'multi_architecture_training/A_train_all_archs_on_small_set'
    architectures = ['Arch_02', 'Arch_03', 'Arch_04', 'Arch_05', 'Arch_07', 'Arch_08', 'Arch_09']
    save_path = os.path.join(base_dir, 'all_archs_initial_config_losses.png')

    # Initialize containers
    train_losses, val_losses = {}, {}
    dummy_train_loss = None
    dummy_val_loss = None

    # Load losses
    for arch in architectures:
        path = os.path.join(base_dir, 'saved_losses', arch, 'losses.pt')
        if os.path.exists(path):
            data = torch.load(path)
            train_losses[arch] = data.get('train_losses', [])
            val_losses[arch] = data.get('val_losses', [])
            # Use dummy from the first available model
            if dummy_train_loss is None and 'dummy_train_loss' in data:
                dummy_train_loss = data['dummy_train_loss']
            if dummy_val_loss is None and 'dummy_val_loss' in data:
                dummy_val_loss = data['dummy_val_loss']

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    # --- TRAIN LOSSES ---
    for arch in architectures:
        ax1.plot(train_losses[arch], label=arch, linewidth=2)
    if dummy_train_loss is not None:
        ax1.axhline(y=dummy_train_loss, color='green', linestyle='--',
                    label='Mean Dummy Train Loss', linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=25)
    ax1.set_ylabel('L1 Loss', fontsize=25)
    ax1.set_title('Training Losses Across Architectures', fontsize=30)
    ax1.tick_params(labelsize=20)
    ax1.grid(True)

    # --- VAL LOSSES ---
    for arch in architectures:
        ax2.plot(val_losses[arch], label=arch, linewidth=2)
    if dummy_val_loss is not None:
        ax2.axhline(y=dummy_val_loss, color='green', linestyle='--',
                    label='Mean Dummy Val Loss', linewidth=2)

    ax2.set_xlabel('Epoch', fontsize=25)
    ax2.set_title('Validation Losses Across Architectures', fontsize=30)
    ax2.tick_params(labelsize=20)
    ax2.grid(True)

    # --- LEGEND ---
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    unique = dict(zip(labels + labels2, handles + handles2))
    fig.legend(unique.values(), unique.keys(), loc='upper center', fontsize=25,
               ncol=max(1, len(unique) // 2), bbox_to_anchor=(0.5, -0.05))

    plt.subplots_adjust(bottom=0.09)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
