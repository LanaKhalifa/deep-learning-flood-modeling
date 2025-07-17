# multi_architecture_training/B_tune_one_arch_on_small_set/plot_losses.py

import os
import torch
import matplotlib.pyplot as plt

def plot_losses():
    base_dir = 'multi_architecture_training/B_tune_one_arch_on_small_set'
    arch_name = 'Arch_04'
    loss_path = os.path.join(base_dir, 'saved_losses', arch_name, 'losses.pt')

    if not os.path.exists(loss_path):
        print(f"⚠️ No loss file found at: {loss_path}")
        return

    data = torch.load(loss_path)
    train_losses = data.get('train_losses', [])
    val_losses = data.get('val_losses', [])
    dummy_train_loss = data.get('dummy_train_loss')
    dummy_val_loss = data.get('dummy_val_loss')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Nimbus Roman']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    if dummy_train_loss is not None:
        ax1.axhline(y=dummy_train_loss, color='green', linestyle='--', label='Dummy Train', linewidth=2)
    ax1.set_title("Training Loss", fontsize=20)
    ax1.set_xlabel("Epoch", fontsize=16)
    ax1.set_ylabel("L1 Loss", fontsize=16)
    ax1.grid(True)
    ax1.legend()

    ax2.plot(val_losses, label='Validation Loss', linewidth=2)
    if dummy_val_loss is not None:
        ax2.axhline(y=dummy_val_loss, color='green', linestyle='--', label='Dummy Val', linewidth=2)
    ax2.set_title("Validation Loss", fontsize=20)
    ax2.set_xlabel("Epoch", fontsize=16)
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(base_dir, 'tuned_arch_04_loss_plot.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()
