# multi_architecture_training/C_train_best_three_on_big_set/plot_all_losses.py

import os
import torch
import matplotlib.pyplot as plt

def plot_all_losses_best_three():
    base_dir = 'multi_architecture_training/C_train_best_three_on_big_set'
    architectures = ['Arch_05', 'Arch_04', 'Arch_07']
    save_path = os.path.join(base_dir, 'best_three_big_losses.png')

    train_losses, val_losses = {}, {}
    dummy_train_loss = None
    dummy_val_loss = None

    for arch in architectures:
        path = os.path.join(base_dir, 'saved_losses', arch, 'losses.pt')
        if os.path.exists(path):
            data = torch.load(path)
            train_losses[arch] = data.get('train_losses', [])
            val_losses[arch] = data.get('val_losses', [])
            if dummy_train_loss is None and 'dummy_train_loss' in data:
                dummy_train_loss = data['dummy_train_loss']
            if dummy_val_loss is None and 'dummy_val_loss' in data:
                dummy_val_loss = data['dummy_val_loss']

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Nimbus Roman']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    for arch in architectures:
        ax1.plot(train_losses[arch], label=arch, linewidth=2)
    if dummy_train_loss is not None:
        ax1.axhline(y=dummy_train_loss, color='green', linestyle='--', label='Dummy Train', linewidth=2)

    ax1.set_title("Training Loss", fontsize=20)
    ax1.set_xlabel("Epoch", fontsize=16)
    ax1.set_ylabel("L1 Loss", fontsize=16)
    ax1.grid(True)
    ax1.legend()

    for arch in architectures:
        ax2.plot(val_losses[arch], label=arch, linewidth=2)
    if dummy_val_loss is not None:
        ax2.axhline(y=dummy_val_loss, color='green', linestyle='--', label='Dummy Val', linewidth=2)

    ax2.set_title("Validation Loss", fontsize=20)
    ax2.set_xlabel("Epoch", fontsize=16)
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()
