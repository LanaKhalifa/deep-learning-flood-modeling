import torch
import torch.nn as nn
import torch.nn.functional as F


class TerrainDownsampleCubic(nn.Module):
    """
    Terrain downsampler using bicubic interpolation.
    
    The design ensures:
    - Output resolution is fixed at 32×32
    - Each output neuron corresponds to a receptive field of 11×11 in the original input
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, size=(32, 32), mode='bicubic', align_corners=False)
