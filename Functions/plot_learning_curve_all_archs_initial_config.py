import os
import pickle
import matplotlib.pyplot as plt

# Define the base directory and architectures
base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/'
architectures = ['Arch_02', 'Arch_03', 'Arch_04', 'Arch_05', 'Arch_07', 'Arch_08', 'Arch_09']
trial_num = 'initial_config'
save_path = os.path.join(base_dir, 'all_archs_initial_config.png')

# Initialize dictionaries to store the losses
train_losses = {}
val_losses = {}
dummy_mean_loss = None

# Load the losses for each architecture
for arch in architectures:
    train_loss_path = os.path.join(base_dir, arch, trial_num, 'losses', 'train', 'G_losses_train.pkl')
    val_loss_path = os.path.join(base_dir, arch, trial_num, 'losses', 'val', 'G_losses_val.pkl')
    dummy_loss_path = os.path.join(base_dir, 'Arch_02', trial_num, 'losses', 'dummy', 'mean_dummy_val_loss.pkl')

    with open(train_loss_path, 'rb') as f:
        train_losses[arch] = pickle.load(f)
    
    with open(val_loss_path, 'rb') as f:
        val_losses[arch] = pickle.load(f)
    
    if dummy_mean_loss is None:
        with open(dummy_loss_path, 'rb') as f:
            dummy_mean_loss = pickle.load(f)

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Plot training losses
for arch in architectures:
    ax1.plot(train_losses[arch], label=f'{arch}_Train Loss')
    
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('L1 Loss', fontsize=14)
ax1.set_title('Training Losses Across Architectures', fontsize=16)
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True)

# Plot validation losses
for arch in architectures:
    ax2.plot(val_losses[arch], label=f'{arch}_Val Loss')
ax2.axhline(y=dummy_mean_loss, color='green', linestyle='--', label='Mean Validation Dummy Loss')
ax2.set_xlabel('Epoch', fontsize=14)
ax2.set_ylabel('L1 Loss', fontsize=14)
ax2.set_title('Validation Losses Across Architectures', fontsize=16)
ax2.legend(loc='upper right', fontsize=12)
ax2.grid(True)

# Save the plot
plt.suptitle(f'All Architectures - Initial Configurations', fontsize=20)
plt.savefig(save_path)
plt.show()
plt.close()
