import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.value_conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.get_back_C = nn.Conv1d(out_channels, in_channels, kernel_size)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        N = H * W
        x_flat = x.view(batch_size, C, N)

        queries = self.query_conv(x_flat)
        keys = self.key_conv(x_flat)
        values = self.value_conv(x_flat)

        attention = torch.bmm(queries.permute(0, 2, 1), keys)
        attention = F.softmax(attention / (C ** 0.5), dim=-1)

        out = torch.bmm(values, attention.permute(0, 2, 1))
        out = self.get_back_C(out).view(batch_size, C, H, W)
        return self.gamma * out + x


class EncoderDecoderWithAttention(nn.Module):
    """
    Encoder-Decoder model with self-attention at the bottleneck.
    Uses large kernels and progressive downsampling followed by transposed convolutions.
    """
    def __init__(self, downsampler):
        super().__init__()
        self.downsampler = downsampler
        self.act = nn.LeakyReLU()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            self.act,
            nn.Conv2d(32, 48, 5, padding=2),
            self.act,
            nn.Conv2d(48, 64, 5, stride=2, padding=2),
            self.act,
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            self.act,
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            self.act
        )

        # Self-Attention
        self.attention = ConvSelfAttention(in_channels=128, out_channels=128)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 69, 3, stride=2, padding=1),
            self.act,
            nn.ConvTranspose2d(69, 64, 3, stride=2, padding=0),
            self.act,
            nn.ConvTranspose2d(64, 48, 4, stride=2, padding=0),
            self.act,
            nn.ConvTranspose2d(48, 32, 5, padding=2),
            self.act,
            nn.ConvTranspose2d(32, 6, 3, padding=1),
            self.act,
            nn.ConvTranspose2d(6, 1, 1),
            nn.LeakyReLU()
        )

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)
        x = self.encoder(x)
        x = self.attention(x)
        return self.decoder(x)
