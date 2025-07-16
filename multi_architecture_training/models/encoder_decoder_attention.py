import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_architecture_training.models.attention import ConvSelfAttention

class EncoderDecoderWithAttention(nn.Module):
    """
    Encoder-Decoder architecture with a self-attention block after the encoder.
    """
    def __init__(self, downsampler):
        super().__init__()
        self.downsampler = downsampler
        self.nonlinearity = nn.LeakyReLU()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            self.nonlinearity,
            nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),
            self.nonlinearity,
            nn.Conv2d(48, 64, kernel_size=5, stride=2, padding=2),
            self.nonlinearity,
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            self.nonlinearity,
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            self.nonlinearity
        )

        # Self-Attention
        self.attention = ConvSelfAttention(in_channels=128, out_channels=128)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 69, kernel_size=3, stride=2, padding=1),
            self.nonlinearity,
            nn.ConvTranspose2d(69, 64, kernel_size=3, stride=2, padding=0),
            self.nonlinearity,
            nn.ConvTranspose2d(64, 48, kernel_size=4, stride=2, padding=0),
            self.nonlinearity,
            nn.ConvTranspose2d(48, 32, kernel_size=5, stride=1, padding=2),
            self.nonlinearity,
            nn.ConvTranspose2d(32, 6, kernel_size=3, stride=1, padding=1),
            self.nonlinearity,
            nn.ConvTranspose2d(6, 1, kernel_size=1),
            nn.LeakyReLU()
        )

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)
        x = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x)
        return x

