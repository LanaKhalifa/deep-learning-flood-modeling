import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ConvSelfAttention  # Assuming you move SelfAttention.py into architectures/attention.py


class NonDownsamplingConvolutions(nn.Module):
    """
    Simple architecture with multiple convolution layers and no spatial downsampling.
    Terrain is passed through a downsampler, concatenated with IC + BC, and fed into stacked convs.
    """
    def __init__(self, downsampler, num_layers=5, num_channels=32, input_channels=3, act='leakyrelu', last_act='leakyrelu'):
        super().__init__()

        if act == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError("Only 'leakyrelu' supported so far.")

        self.downsampler = downsampler

        layers = []
        # Initial conv layer
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1))
        layers.append(self.activation)

        # Intermediate layers
        for _ in range(1, num_layers - 1):
            layers.append(nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1))
            layers.append(self.activation)

        # Final output layer
        layers.append(nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, stride=1, padding=1))
        if last_act == 'leakyrelu':
            layers.append(nn.LeakyReLU())  # usually safe to include for difference prediction

        self.conv_net = nn.Sequential(*layers)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)
        return self.conv_net(x)



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




class NonDownsamplingConvolutionsWithAttention(nn.Module):
    """
    Same as NonDownsamplingConvolutions but includes a ConvSelfAttention layer in the middle.
    """
    def __init__(self, downsampler, num_layers=6, num_channels=32, input_channels=3, act='leakyrelu', last_act='leakyrelu', num_attentions=1):
        super().__init__()
        self.downsampler = downsampler

        if act == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError("Only 'leakyrelu' supported so far.")

        layers = []

        # Initial layer
        layers.append(nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1))
        layers.append(self.activation)

        # First half of convolutional layers
        for _ in range(1, num_layers // 2):
            layers.append(nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1))
            layers.append(self.activation)

        # Self-attention layer
        for _ in range(num_attentions):
            layers.append(ConvSelfAttention(num_channels))

        # Second half of convolutional layers
        for _ in range(num_layers // 2, num_layers - 1):
            layers.append(nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1))
            layers.append(self.activation)

        # Final conv + activation
        layers.append(nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, stride=1, padding=1))
        if last_act == 'leakyrelu':
            layers.append(nn.LeakyReLU())

        self.conv_net = nn.Sequential(*layers)

    def forward(self, terrain, depths):
        if self.downsampler is not None:
            terrain_out = self.downsampler(terrain)
            x = torch.cat([terrain_out, depths], dim=1)
        else:
            x = depths
        return self.conv_net(x)

class ClassicUNet(nn.Module):
    """
    Classic UNet architecture as proposed by Ronneberger et al. (2015).
    Encoder-decoder structure with skip connections and max pooling.
    Terrain is downsampled, concatenated with IC + BC, then passed through the network.
    """
    def __init__(self, downsampler):
        super().__init__()
        self.downsampler = downsampler
        self.act = nn.LeakyReLU()

        # Encoder
        self.e11 = nn.Conv2d(3, 64, 3, padding=1)
        self.e12 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.e21 = nn.Conv2d(64, 128, 3, padding=1)
        self.e22 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.e31 = nn.Conv2d(128, 256, 3, padding=1)
        self.e32 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.e41 = nn.Conv2d(256, 512, 3, padding=1)
        self.e42 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.e51 = nn.Conv2d(512, 1024, 3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, 3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, 3, padding=1)
        self.d12 = nn.Conv2d(512, 512, 3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d21 = nn.Conv2d(512, 256, 3, padding=1)
        self.d22 = nn.Conv2d(256, 256, 3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d31 = nn.Conv2d(256, 128, 3, padding=1)
        self.d32 = nn.Conv2d(128, 128, 3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d41 = nn.Conv2d(128, 64, 3, padding=1)
        self.d42 = nn.Conv2d(64, 64, 3, padding=1)

        self.outconv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)

        # Encoder
        xe1 = self.act(self.e11(x))
        xe1 = self.act(self.e12(xe1))
        xp1 = self.pool1(xe1)

        xe2 = self.act(self.e21(xp1))
        xe2 = self.act(self.e22(xe2))
        xp2 = self.pool2(xe2)

        xe3 = self.act(self.e31(xp2))
        xe3 = self.act(self.e32(xe3))
        xp3 = self.pool3(xe3)

        xe4 = self.act(self.e41(xp3))
        xe4 = self.act(self.e42(xe4))
        xp4 = self.pool4(xe4)

        xe5 = self.act(self.e51(xp4))
        xe5 = self.act(self.e52(xe5))

        # Decoder
        x = self.upconv1(xe5)
        x = torch.cat([x, xe4], dim=1)
        x = self.act(self.d11(x))
        x = self.act(self.d12(x))

        x = self.upconv2(x)
        x = torch.cat([x, xe3], dim=1)
        x = self.act(self.d21(x))
        x = self.act(self.d22(x))

        x = self.upconv3(x)
        x = torch.cat([x, xe2], dim=1)
        x = self.act(self.d31(x))
        x = self.act(self.d32(x))

        x = self.upconv4(x)
        x = torch.cat([x, xe1], dim=1)
        x = self.act(self.d41(x))
        x = self.act(self.d42(x))

        out = self.act(self.outconv(x))
        return out



class ConvSelfAttention(nn.Module):
    """
    Self-Attention mechanism applied over flattened spatial dimensions.
    """
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

