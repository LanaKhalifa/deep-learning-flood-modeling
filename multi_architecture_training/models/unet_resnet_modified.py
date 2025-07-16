import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockType01(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.selu = nn.SELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.selu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return self.selu(out)


class ResidualBlockType02(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.selu = nn.SELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
    
    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.selu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return self.selu(out)


class BicubicUpsampling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)


class UNetResNetModified(nn.Module):
    def __init__(self, downsampler):
        super().__init__()
        self.downsampler = downsampler

        self.res_block1 = ResidualBlockType01(3, 15)
        self.res_block2 = ResidualBlockType02(15, 30)
        self.res_block3 = ResidualBlockType02(30, 60)
        self.res_block4 = ResidualBlockType02(60, 120)

        self.upsample1 = BicubicUpsampling()
        self.res_block5 = ResidualBlockType01(120, 60)
        self.upsample2 = BicubicUpsampling()
        self.res_block6 = ResidualBlockType01(60, 30)
        self.upsample3 = BicubicUpsampling()
        self.res_block7 = ResidualBlockType01(30, 15)
        self.res_block8 = ResidualBlockType01(15, 15)

        self.final_conv = nn.Conv2d(15, 1, kernel_size=1)
        self.activation = nn.LeakyReLU()

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        x = self.upsample1(x)
        x = self.res_block5(x)
        x = self.upsample2(x)
        x = self.res_block6(x)
        x = self.upsample3(x)
        x = self.res_block7(x)
        x = self.res_block8(x)

        x = self.final_conv(x)
        return self.activation(x)

