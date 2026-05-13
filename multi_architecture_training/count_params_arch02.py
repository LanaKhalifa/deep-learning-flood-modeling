#!/usr/bin/env python
"""Count parameters for Arch_02 (Non-downsampling Convolutions, ablation), excluding downsampler.
Run with: conda activate flood_env && PYTHONPATH=. python multi_architecture_training/count_params_arch02.py
"""
from multi_architecture_training.models.non_downsampling_convolutions import (
    NonDownsamplingConvolutions,
)

# Stage B config (no downsampler)
model = NonDownsamplingConvolutions(
    downsampler=None,
    arch_num_layers=12,
    arch_num_c=32,
    arch_input_c=3,
)

total = sum(p.numel() for p in model.conv_net.parameters())
trainable = sum(p.numel() for p in model.conv_net.parameters() if p.requires_grad)
size_mb = total * 4 / 1e6  # float32

print("Non-downsampling Convolutions (Arch_02, ablation)")
print("Config: arch_num_layers=12, arch_num_c=32")
print("Parameters (conv_net only, excluding downsampler):", total)
print("Trainable:", trainable)
print("Size (MB, float32):", round(size_mb, 4))
