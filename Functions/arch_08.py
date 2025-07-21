import torch 
import torch.nn as nn 
import torch.nn.functional as F

#%% ResidualBlockType01    
class ResidualBlockType01(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockType01, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.selu = nn.SELU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.residual_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.Leaky = nn.LeakyReLU()
    def forward(self, x):
        x_in = x
        x = self.conv1(x)
        x = self.selu(x)
        x = self.conv2(x)

        x_residual = self.residual_conv(x_in)

        x_out = x + x_residual
        return self.selu(x_out)
    
#%% ResidualBlockType02    
class ResidualBlockType02(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockType02, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.selu = nn.SELU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.residual_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        x_in = x
        x = self.conv1(x)
        x = self.selu(x)
        x = self.conv2(x)

        x_residual = self.residual_conv(x_in)

        x_out = x + x_residual
        return self.selu(x_out)
#%% BicubicUpsampling
class BicubicUpsampling(nn.Module):
    def __init__(self):
        super(BicubicUpsampling, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=True)

#%% Full Model
class arch_08(nn.Module):
    def __init__(self, downsampler):
        super(arch_08, self).__init__()
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
        self.final_conv = nn.Conv2d(in_channels=15, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()
        self.Leaky = nn.LeakyReLU()

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
        x = self.Leaky(x)
        return x