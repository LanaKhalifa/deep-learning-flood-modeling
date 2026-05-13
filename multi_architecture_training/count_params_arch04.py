#!/usr/bin/env python
"""Count parameters for Arch_04 (Non-downsampling with Self-Attention), excluding downsampler.
Run with: conda activate flood_env && python multi_architecture_training/count_params_arch04.py
"""
from multi_architecture_training.models.non_downsampling_convolutions_attention import (
    NonDownsamplingConvolutionsWithAttention,
)

# Stage B config (no downsampler)
model = NonDownsamplingConvolutionsWithAttention(
    downsampler=None,
    arch_num_layers=12,
    arch_num_c=32,
    arch_input_c=3,
    arch_num_attentions=2,
)

total = sum(p.numel() for p in model.conv_net.parameters())
trainable = sum(p.numel() for p in model.conv_net.parameters() if p.requires_grad)
size_mb = total * 4 / 1e6  # float32

print("Non-downsampling Convolutions with Self-Attention (Arch_04)")
print("Config: arch_num_layers=12, arch_num_c=32, arch_num_attentions=2")
print("Parameters (conv_net only, excluding downsampler):", total)
print("Trainable:", trainable)
print("Size (MB, float32):", round(size_mb, 4))
