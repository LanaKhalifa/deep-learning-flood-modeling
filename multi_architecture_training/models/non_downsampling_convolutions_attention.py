import torch
import torch.nn as nn
from multi_architecture_training.models.attention import ConvSelfAttention 


class NonDownsamplingConvolutionsWithAttention(nn.Module):
    """
    Simple encoder-style model with stacked conv layers and attention layers distributed evenly.
    """
    def __init__(self, downsampler, arch_num_layers=6, arch_num_c=32, arch_input_c=3, arch_act='leakyrelu', arch_last_act='leakyrelu', arch_num_attentions=2):
        super().__init__()

        if arch_act == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError("Only 'leakyrelu' is currently supported.")

        self.downsampler = downsampler

        layers = [
            nn.Conv2d(arch_input_c, arch_num_c, kernel_size=3, stride=1, padding=1),
            self.activation
        ]

        # Calculate positions for attention layers (distributed evenly)
        attention_positions = []
        if arch_num_attentions > 0:
            # Place attention layers at equal intervals
            interval = (arch_num_layers - 1) / (arch_num_attentions + 1)
            for i in range(arch_num_attentions):
                position = int((i + 1) * interval)
                attention_positions.append(position)

        # Build the network with attention layers at calculated positions
        attention_idx = 0
        for layer_idx in range(1, arch_num_layers - 1):
            layers += [
                nn.Conv2d(arch_num_c, arch_num_c, kernel_size=3, stride=1, padding=1),
                self.activation
            ]
            
            # Insert attention layer if this is an attention position
            if attention_idx < len(attention_positions) and layer_idx == attention_positions[attention_idx]:
                layers.append(ConvSelfAttention(arch_num_c, arch_num_c))
                attention_idx += 1

        # Final conv
        layers.append(nn.Conv2d(arch_num_c, 1, kernel_size=3, stride=1, padding=1))
        if arch_last_act == 'leakyrelu':
            layers.append(nn.LeakyReLU())

        self.conv_net = nn.Sequential(*layers)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain) if self.downsampler else depths
        x = torch.cat([terrain_out, depths], dim=1)
        return self.conv_net(x)
