import torch
import torch.nn as nn
import torch.nn.functional as F


class AlternatingStrideConv2d(nn.Module):
    """
    Custom 2D convolution that extracts patches with a larger hop (stride1 + stride2),
    applies a Conv2d over each patch, and reconstructs the output spatially.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride1=2, stride2=3, padding=0):
        super().__init__()
        self.hop = stride1 + stride2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride1, padding=padding)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        output_height = (height // self.hop) * 2
        output_width = (width // self.hop) * 2

        # Extract patches
        patches = F.unfold(x, kernel_size=(self.hop, self.hop), stride=self.hop)
        patches = patches.view(batch_size, channels, self.hop, self.hop, -1)
        patches = patches.permute(0, 4, 1, 2, 3).contiguous().view(-1, channels, self.hop, self.hop)

        # Apply convolution to each patch
        output_patches = self.conv(patches)

        # Reshape back to image format
        output_patches = output_patches.view(batch_size, -1, self.conv.out_channels, 2, 2)
        output_patches = output_patches.view(batch_size, output_height // 2, output_width // 2, self.conv.out_channels, 2, 2)
        output_patches = output_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        output = output_patches.view(batch_size, self.conv.out_channels, output_height, output_width)

        return output
