import torch.nn as nn 
import torch.nn.functional as F
import torch

class AlternatingStrideConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride1=2, stride2=3, padding=0):
        super().__init__()  # Corrected super() call
        self.hop = stride1 + stride2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride1, padding=padding)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        output_height = height // self.hop * 2
        output_width = width // self.hop * 2

        patches = F.unfold(x, kernel_size=(self.hop, self.hop), stride=self.hop) # tested
        patches = patches.view(batch_size, channels, self.hop, self.hop, -1) # tested
        patches = patches.permute(0, 4, 1, 2, 3).contiguous().view(-1, channels, self.hop, self.hop) # tested

        # Apply the| convolution to all patches
        output_patches = self.conv(patches) # tested

        # Reshape the output patches to the output shape
        output_patches = output_patches.view(batch_size, -1, self.conv.out_channels, 2, 2) # tested only visually 
        output_patches = output_patches.view(batch_size, output_height // 2, output_width // 2, self.conv.out_channels, 2, 2) # tested only visually
        output_patches = output_patches.permute(0, 3, 1, 4, 2, 5).contiguous() # tested only visually
        output = output_patches.view(batch_size, self.conv.out_channels, output_height, output_width) # tested only visually 

        return output
    
    
