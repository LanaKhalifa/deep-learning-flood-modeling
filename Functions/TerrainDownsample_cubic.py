import torch.nn as nn 
import torch.nn.functional as F


class TerrainDownsample_cubic(nn.Module):
    """
    Terrain downsampling is built such that:
    * The receptive field of each neuron in the output layer is exactly 11x11
    * Uses cubic interpolation to downsample the input from 321x321 to 32x32.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x = F.interpolate(x, size=(32, 32), mode='bicubic', align_corners=False)
        return x