import torch
import torch.nn.functional as F
import torch.nn as nn
 
# Sample input tensor
x = torch.arange(1, 5 * 3 * 10 * 10 + 1).view(5, 3, 10, 10).double()

# Parameters
batch_size, channels, height, width = x.size()
stride1 = 2
stride2 = 3
hop = stride1 + stride2

# Unfold
unfolded = F.unfold(x, kernel_size=(hop, hop), stride=hop)
print("Unfolded shape:", unfolded.shape)  # Expected shape: (N, C * K_H * K_W, L)

# Reshape patches
patches = unfolded.view(batch_size, channels, hop, hop, -1)
print("Reshaped patches shape:", patches.shape)  # Expected shape: (N, C, K_H, K_W, L)

# Permute 
patches = patches.permute(0, 4, 1, 2, 3).contiguous()
print("Permuted patches shape:", patches.shape)  # Expected shape: (N, L, C, K_H, K_W)

# View
patches = patches.view(-1, channels, hop, hop)
print("View patches shape:", patches.shape)  # Expected shape: (N*L, C, K_H, K_W)

# Conv layer with random weights
conv = nn.Conv2d(channels, 2, 3, stride=2, padding=0).double()

# Manually set the weights and biases from a normal distribution
with torch.no_grad():
    conv.weight.normal_(mean=10, std=5)  # Initialize weights from a normal distribution
    conv.bias.normal_(mean=1, std=0)    # Initialize biases from a normal distribution

# Disable gradient updates
conv.weight.requires_grad = False
conv.bias.requires_grad = False
 
# Apply convolution
output_patches = conv(patches)

# Extracting patches manually for comparison
s=0
patches_manual = []
patches_manual_no_conv = []
for n in range(batch_size):
    for i in range(0, height, hop):  # Ensure correct patch extraction
        for j in range(0, width, hop):  # Ensure correct patch extraction
            patch = x[n, :, i:i+hop, j:j+hop]
            print('MANUAL')
            print(patch[0])
            print('FROMBEFORE')
            print(patches[s, 0, :, :])
            patches_manual_no_conv.append(patch)
            print('MANUAL')
            print(conv(patch)[0])
            print('FROMBEFORE')
            print(output_patches[s,0,:,:])
            patches_manual.append(conv(patch))
            s = s +1

# Convert list to tensor
patches_manual = torch.stack(patches_manual, dim=0)
patches_manual_no_conv = torch.stack(patches_manual_no_conv, dim=0)

# Compare tensors
print(torch.sum(output_patches - patches_manual))
print(torch.sum(patches - patches_manual_no_conv))
