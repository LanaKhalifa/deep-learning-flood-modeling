import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoderWithLargeConvolutions(nn.Module):
    """
    Encoder-decoder architecture using large kernel convolutions and strides
    to aggressively downsample and upsample the feature maps.
    """
    def __init__(self, downsampler):
        super().__init__()
        self.downsampler = downsampler

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=6, stride=1, padding=0)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=6, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=11, stride=11, padding=0)
        self.tanh = nn.Tanh()

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=32, kernel_size=11, stride=11, padding=0)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=8, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=6, stride=1, padding=0)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)

        # Encoder
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.conv3(x)
        x = self.tanh(x)

        # Decoder
        x = self.deconv1(x)
        x = self.tanh(x)
        x = self.deconv2(x)
        x = self.tanh(x)
        x = self.deconv3(x)
        x = self.tanh(x)

        return x

