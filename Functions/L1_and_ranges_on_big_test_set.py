import torch
import numpy as np
import matplotlib.pyplot as plt

from TerrainDownsample_k11s1p0 import TerrainDownsample_k11s1p0
from arch_04 import arch_04
import os

#%% settings constant
# Set the default tensor type to float64
torch.set_default_dtype(torch.float64) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% settings
trial_num = 'trial_final'
Architecture_num = 'Arch_04'
prj_num = '_'

BC_thickness = 2
num_epochs = 2
weight_init = 'kaiming'
initial_lr = 0.001

terrain_c = 1

down_c_start = 1
down_c1 = 10
down_c2 = 20
down_c_end = 1
down_act = 'leakyrelu'
down_type = 'k11s10'

arch_num_attentions = 1
arch_num_layers = 6
arch_num_c = 32 
input_c = 2
arch_input_c = down_c_end +  input_c
arch_act = 'leakyrelu'
arch_last_act = 'leakyrelu'

base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final'
dataloaders_dir = os.path.join(base_dir, 'Dataloaders', prj_num)
model_dir = os.path.join(base_dir, f'{Architecture_num}/{trial_num}/trained_models')
model_path = os.path.join(model_dir, 'model.pth')

# Load the trained model
shared_terrain_downsample = TerrainDownsample_k11s1p0(down_c_start, down_c_end, down_c1, down_c2).to(device)
netG = arch_04(shared_terrain_downsample, arch_num_layers, arch_num_c, arch_input_c, arch_act, arch_last_act, arch_num_attentions).to(device)
netG.load_state_dict(torch.load(model_path))
netG.eval()  # Set the model to evaluation mode
print("Trained model loaded successfully.")

# Load the test data loader
loader_path = os.path.join(dataloaders_dir, 'big_test_loader_Terrain_1_BC_2.pt')
loader = torch.load(loader_path)
print("big_test loader loaded successfully.")

# Initialize dictionaries to store data for each group
group_losses = {'group_1': 0, 'group_2': 0, 'group_3': 0, 'group_4': 0}
group_counts = {'group_1': 0, 'group_2': 0, 'group_3': 0, 'group_4': 0}
group_samples = {'group_1': [], 'group_2': [], 'group_3': [], 'group_4': []}

# Step 1: Group samples based on min and max pixel values of the labels
for terrains, data, labels in loader:
    max_vals = labels.max(dim=3)[0].max(dim=2)[0]  # Maximum pixel value in each label # must think whether this is ok 
    min_vals = labels.min(dim=3)[0].min(dim=2)[0]  # Minimum pixel value in each label # must think whether this is ok 

    for i in range(labels.size(0)):  # Iterate over batch
        max_val, min_val = max_vals[i].item(), min_vals[i].item()

        if max_val < 1:
            group = 'group_1'
        elif min_val > 1 and max_val < 2:
            group = 'group_2'
        elif min_val > 4 and max_val < 6:
            group = 'group_3'
        elif min_val > 6 and max_val < 10:
            group = 'group_4'
        else:
            continue  # Skip samples that don't fall into any group

        # Accumulate the samples into the respective group
        group_samples[group].append((terrains[i:i+1], data[i:i+1], labels[i:i+1]))

# Step 2 and 3: Pass each group through the model and calculate the L1 loss
with torch.no_grad():
    for group, samples in group_samples.items():
        if not samples:
            continue

        terrains_group = torch.cat([s[0] for s in samples], dim=0).to(device)
        data_group = torch.cat([s[1] for s in samples], dim=0).to(device)
        labels_group = torch.cat([s[2] for s in samples], dim=0).to(device)

        # Model prediction
        y_fake_group = netG(terrains_group, data_group)

        # Calculate L1 loss
        l1_loss_group = torch.abs(y_fake_group - labels_group).mean().item()

        # Store the loss and count for the group
        group_losses[group] = l1_loss_group
        group_counts[group] = labels_group.size(0)

# Step 4 and 5: Plotting the results
groups = ['group_1', 'group_2', 'group_3', 'group_4']
loss_values = [group_losses[group] for group in groups]
sample_counts = [group_counts[group] for group in groups]

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Groups')
ax1.set_ylabel('L1 Loss', color=color)
ax1.bar(groups, loss_values, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:orange'
ax2.set_ylabel('Number of Samples', color=color)  # we already handled the x-label with ax1
ax2.plot(groups, sample_counts, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('L1 Loss and Sample Counts per Group')
plt.show()

print("L1 Loss and Sample Counts per Group calculation completed.")
