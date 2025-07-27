
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSelfAttention(nn.Module):
    """
    Applies self-attention over the spatial features by reshaping the input and applying Conv1d operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.value_conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.get_back_C = nn.Conv1d(out_channels, in_channels, kernel_size)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        N = width * height

        x_flat = x.view(batch_size, C, N)
        queries = self.query_conv(x_flat)
        keys = self.key_conv(x_flat)
        values = self.value_conv(x_flat)

        attention = torch.bmm(queries.permute(0, 2, 1), keys)
        attention = F.softmax(attention / (C ** 0.5), dim=-1)

        out = torch.bmm(values, attention.permute(0, 2, 1))
        out_with_C = self.get_back_C(out).view(batch_size, C, width, height)

        return self.gamma * out_with_C + x
