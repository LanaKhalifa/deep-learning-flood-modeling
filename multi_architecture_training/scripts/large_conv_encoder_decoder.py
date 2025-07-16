import torch
import torch.nn as nn

class LargeConvEncoderDecoder(nn.Module):
    """
    Encoder-Decoder architecture using large convolution kernels and strides.
    """

    def __init__(self, downsampler):
        super().__init__()
        self.downsampler = downsampler
        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

        # Encoder
        self.conv1 = nn.Conv2d(3, 4, kernel_size=6, stride=1)
        self.conv2 = nn.Conv2d(4, 32, kernel_size=6, stride=1)
        self.conv3 = nn.Conv2d(32, 256, kernel_size=11, stride=11)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(256, 32, kernel_size=11, stride=11)
        self.deconv2 = nn.ConvTranspose2d(32, 3, kernel_size=8, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(3, 1, kernel_size=6, stride=1)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)

        # Encoder
        x = self.prelu(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.tanh(self.conv3(x))

        # Decoder
        x = self.tanh(self.deconv1(x))
        x = self.tanh(self.deconv2(x))
        x = self.tanh(self.deconv3(x))

        return x
