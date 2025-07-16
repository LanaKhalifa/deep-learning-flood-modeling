import torch
import torch.nn as nn
from .attention import ConvSelfAttention

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

        layers = [
            nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),
            self.activation
        ]

        for _ in range(1, num_layers - 1):
            layers += [
                nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
                self.activation
            ]

        layers.append(nn.Conv2d(num_channels, 1, kernel_size=3, stride=1, padding=1))
        if last_act == 'leakyrelu':
            layers.append(nn.LeakyReLU())

        self.conv_net = nn.Sequential(*layers)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)
        return self.conv_net(x)


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

        layers = [
            nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=1, padding=1),
            self.activation
        ]

        for _ in range(1, num_layers // 2):
            layers += [
                nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
                self.activation
            ]

        for _ in range(num_attentions):
            layers.append(ConvSelfAttention(num_channels))

        for _ in range(num_layers // 2, num_layers - 1):
            layers += [
                nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
                self.activation
            ]

        layers.append(nn.Conv2d(num_channels, 1, kernel_size=3, stride=1, padding=1))
        if last_act == 'leakyrelu':
            layers.append(nn.LeakyReLU())

        self.conv_net = nn.Sequential(*layers)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain) if self.downsampler else depths
        x = torch.cat([terrain_out, depths], dim=1)
        return self.conv_net(x)
