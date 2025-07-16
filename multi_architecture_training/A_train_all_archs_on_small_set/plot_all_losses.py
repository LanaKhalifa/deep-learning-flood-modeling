import os
import torch
import matplotlib.pyplot as plt

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(__file__)  # This script's directory
LOSSES_DIR = os.path.join(BASE_DIR, "saved_losses")
OUTPUT_PATH = os.path.join(BASE_DIR, "all_archs_learning_curves.png")
ARCHS = sorted(os.listdir(LOSSES_DIR))  # Automatically detect trained archs

# === PLOTTING ===
def plot_loss_curves(losses_dir, archs, output_path):
    plt.figure(figsize=(16, 9))

    for arch in archs:
        loss_file = os.path.join(losses_dir, arch, "losses.pt")
        if not os.path.isfile(loss_file):
            print(f"[!] Skipping {arch}: No losses.pt found.")
            continue

        data = torch.load(loss_file)
        train_losses = data['train_losses']
        val_losses = data['val_losses']

        plt.plot(train_losses, label=f"{arch} - Train", linestyle='-')
        plt.plot(val_losses, label=f"{arch} - Val", linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.title("Training & Validation Loss Across Architectures")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"[✓] Saved plot to {output_path}")


# === ENTRY POINT ===
if __name__ == "__main__":
    plot_loss_curves(LOSSES_DIR, ARCHS, OUTPUT_PATH)
