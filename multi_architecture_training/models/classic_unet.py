
import torch
import torch.nn as nn


class ClassicUNet(nn.Module):
    """
    A classic UNet-style encoder-decoder architecture with skip connections.
    """
    def __init__(self, downsampler):
        super().__init__()
        self.downsampler = downsampler
        self.act = nn.LeakyReLU()

        # Encoder
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, 1, kernel_size=1)
        self.tanh = nn.Tanh()  # Optional: replace act here if needed

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)

        # Encoder
        xe11 = self.act(self.e11(x))
        xe12 = self.act(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = self.act(self.e21(xp1))
        xe22 = self.act(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = self.act(self.e31(xp2))
        xe32 = self.act(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = self.act(self.e41(xp3))
        xe42 = self.act(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = self.act(self.e51(xp4))
        xe52 = self.act(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.act(self.d11(xu11))
        xd12 = self.act(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.act(self.d21(xu22))
        xd22 = self.act(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.act(self.d31(xu33))
        xd32 = self.act(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.act(self.d41(xu44))
        xd42 = self.act(self.d42(xd41))

        out = self.act(self.outconv(xd42))
        return out
