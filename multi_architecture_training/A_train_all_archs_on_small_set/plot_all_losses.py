import os
import torch
import matplotlib.pyplot as plt

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(__file__)  # Directory of this script
LOSSES_DIR = os.path.join(BASE_DIR, "saved_losses")
OUTPUT_PATH = os.path.join(BASE_DIR, "all_archs_learning_curves.png")
ARCHS = sorted(os.listdir(LOSSES_DIR))  # Automatically detect trained architectures

# === FONT SETTINGS ===
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman']


def plot_loss_curves(losses_dir, arch_names, save_path):
    train_losses = {}
    val_losses = {}

    for arch in arch_names:
        loss_path = os.path.join(losses_dir, arch, "losses.pt")
        if not os.path.exists(loss_path):
            continue
        data = torch.load(loss_path)
        train_losses[arch] = data.get("train_losses", [])
        val_losses[arch] = data.get("val_losses", [])

    # === PLOTTING ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    for arch in arch_names:
        if arch in train_losses:
            ax1.plot(train_losses[arch], label=arch, linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=25, labelpad=20)
    ax1.set_ylabel("L1 Loss", fontsize=25, labelpad=20)
    ax1.set_title("Training Losses Across Architectures", fontsize=30, pad=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.grid(True)

    for arch in arch_names:
        if arch in val_losses:
            ax2.plot(val_losses[arch], label=arch, linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=25, labelpad=20)
    ax2.set_title("Validation Losses Across Architectures", fontsize=30, pad=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.grid(True)

    # Combine legends without duplication
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    seen = set()
    unique_handles, unique_labels = [], []

    for h, l in zip(handles1 + handles2, labels1 + labels2):
        if l not in seen:
            unique_handles.append(h)
            unique_labels.append(l)
            seen.add(l)

    fig.legend(unique_handles, unique_labels, loc="upper center", fontsize=25,
               ncol=max(1, len(unique_labels) // 2), bbox_to_anchor=(0.5, -0.05))
    plt.subplots_adjust(bottom=0.09)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_all_arch_losses():
    plot_loss_curves(LOSSES_DIR, ARCHS, OUT_
