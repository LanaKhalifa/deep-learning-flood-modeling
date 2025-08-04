import torch
import torch.nn as nn


class NonDownsamplingConvolutions(nn.Module):
    """
    Simple architecture with multiple convolution layers and no spatial downsampling.
    Terrain is passed through a downsampler, concatenated with IC + BC, and fed into stacked convs.
    """
    def __init__(self, downsampler, arch_num_layers=5, arch_num_c=32, arch_input_c=3, arch_act='leakyrelu', arch_last_act='leakyrelu'):
        super().__init__()

        if arch_act == 'leakyrelu':
            self.nonlinearity = nn.LeakyReLU()
        else:
            raise NotImplementedError("Only 'leakyrelu' is currently supported.")

        self.downsampler = downsampler

        layers = [
            nn.Conv2d(in_channels=arch_input_c, out_channels=arch_num_c, kernel_size=3, stride=1, padding=1),
            self.nonlinearity
        ]

        for _ in range(1, arch_num_layers - 1):
            layers += [
                nn.Conv2d(in_channels=arch_num_c, out_channels=arch_num_c, kernel_size=3, stride=1, padding=1),
                self.nonlinearity
            ]

        layers.append(nn.Conv2d(in_channels=arch_num_c, out_channels=1, kernel_size=3, stride=1, padding=1))

        if arch_last_act == 'leakyrelu':
            layers.append(nn.LeakyReLU())

        self.conv_net = nn.Sequential(*layers)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)
        return self.conv_net(x)
