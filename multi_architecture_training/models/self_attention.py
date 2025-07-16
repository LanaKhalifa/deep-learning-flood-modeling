import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSelfAttention(nn.Module):
    """
    Convolutional self-attention module that reshapes feature maps into sequences
    and applies attention in the flattened spatial domain.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.get_back_C = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        N = width * height

        # Flatten spatial dimensions: B x C x (W*H)
        x_flat = x.view(batch_size, C, N)

        queries = self.query_conv(x_flat)   # B x C x N
        keys    = self.key_conv(x_flat)     # B x C x N
        values  = self.value_conv(x_flat)   # B x C x N

        attention = torch.bmm(queries.permute(0, 2, 1), keys)  # B x N x N
        attention = F.softmax(attention / (C ** 0.5), dim=-1)

        out = torch.bmm(values, attention.permute(0, 2, 1))  # B x C x N
        out_with_C = self.get_back_C(out).view(batch_size, C, width, height)

        return self.gamma * out_with_C + x
