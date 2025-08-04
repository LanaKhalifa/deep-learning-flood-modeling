import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_architecture_training.models.alternating_stride_conv2d import AlternatingStrideConv2d 


class TerrainDownsampleAlternating(nn.Module):
    """
    Terrain downsampler using a combination of standard and alternating stride convolutions.
    """
    def __init__(self, c_start=1, c1=1, c2=1, c_end=1):
        super().__init__()
        self.c1 = nn.Conv2d(c_start, c1, kernel_size=3, stride=2, padding=0)
        self.c2 = AlternatingStrideConv2d(c1, c2, kernel_size=3, stride1=2, stride2=3, padding=0)
        self.c3 = nn.Conv2d(c2, c_end, kernel_size=2, stride=2, padding=0)

        self.nonlinearity = nn.LeakyReLU()

    def forward(self, x):
        x = self.nonlinearity(self.c1(x))
        x = self.nonlinearity(self.c2(x))
        x = self.nonlinearity(self.c3(x))
        return x
