import torch
import torch.nn as nn

class DownConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)

class UpConv(nn.Module):
    def __init__(self, in_c, out_c, last=False):
        super().__init__()
        if last:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU()
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            )

    def forward(self, x):
        return self.block(x)

class SimplifiedUNet(nn.Module):
    """
    UNet-style encoder-decoder architecture based on Alhada-Lahbabi et al. (2023).
    Terrain is downsampled and concatenated with IC+BC before passing through the network.
    """
    def __init__(self, downsampler, num_channels=[3, 32, 64, 128, 256, 256, 256, 128, 64, 32, 1]):
        super().__init__()
        self.downsampler = downsampler

        # Encoder
        self.down1 = DownConv(num_channels[0], num_channels[1])
        self.down2 = DownConv(num_channels[1], num_channels[2])
        self.down3 = DownConv(num_channels[2], num_channels[3])
        self.down4 = DownConv(num_channels[3], num_channels[4])
        self.down5 = DownConv(num_channels[4], num_channels[5])

        # Decoder
        self.up1 = UpConv(num_channels[5], num_channels[6])
        self.up2 = UpConv(num_channels[6]*2, num_channels[7])
        self.up3 = UpConv(num_channels[7]*2, num_channels[8])
        self.up4 = UpConv(num_channels[8]*2, num_channels[9])
        self.up5 = UpConv(num_channels[9]*2, num_channels[10], last=True)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)

        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        # Decoder with skip connections
        x6 = self.up1(x5)
        x6 = torch.cat([x6, x4], dim=1)
        x7 = self.up2(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.up3(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x9 = self.up4(x8)
        x9 = torch.cat([x9, x1], dim=1)
        x10 = self.up5(x9)
        return x10
