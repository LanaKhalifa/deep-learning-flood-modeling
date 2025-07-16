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
    def __init__(self, in_c, out_c, is_last=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU() if is_last else nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class SimplifiedUNet(nn.Module):
    def __init__(self, downsampler, num_c_encoder=[3, 32, 64, 128, 256, 256, 256, 128, 64, 32, 1]):
        super().__init__()
        self.downsampler = downsampler

        # Encoder
        self.down_conv_1 = DownConv(num_c_encoder[0], num_c_encoder[1])
        self.down_conv_2 = DownConv(num_c_encoder[1], num_c_encoder[2])
        self.down_conv_3 = DownConv(num_c_encoder[2], num_c_encoder[3])
        self.down_conv_4 = DownConv(num_c_encoder[3], num_c_encoder[4])
        self.down_conv_5 = DownConv(num_c_encoder[4], num_c_encoder[5])

        # Decoder
        self.up_conv_1 = UpConv(num_c_encoder[5], num_c_encoder[6])
        self.up_conv_2 = UpConv(num_c_encoder[6] + num_c_encoder[4], num_c_encoder[7])
        self.up_conv_3 = UpConv(num_c_encoder[7] + num_c_encoder[3], num_c_encoder[8])
        self.up_conv_4 = UpConv(num_c_encoder[8] + num_c_encoder[2], num_c_encoder[9])
        self.up_conv_5 = UpConv(num_c_encoder[9] + num_c_encoder[1], num_c_encoder[10], is_last=True)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)

        # Encoder
        x1 = self.down_conv_1(x)
        x2 = self.down_conv_2(x1)
        x3 = self.down_conv_3(x2)
        x4 = self.down_conv_4(x3)
        x5 = self.down_conv_5(x4)

        # Decoder with skip connections
        x6 = self.up_conv_1(x5)
        x6 = torch.cat([x6, x4], dim=1)
        x7 = self.up_conv_2(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.up_conv_3(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x9 = self.up_conv_4(x8)
        x9 = torch.cat([x9, x1], dim=1)
        x10 = self.up_conv_5(x9)

        return x10
