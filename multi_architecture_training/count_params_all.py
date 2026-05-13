#!/usr/bin/env python
"""Count parameters for all Stage B architectures, excluding downsampler (main network only).
Run from repo root: conda activate flood_env && PYTHONPATH=. python multi_architecture_training/count_params_all.py
"""
import sys
sys.path.insert(0, ".")

from config.model_configs import get_stage_B_configs

configs = get_stage_B_configs()
results = []

for arch_name, config in configs.items():
    model_class = config["model_class"]
    params = config.get("params", {})
    downsampler_class = config["downsampler_class"]
    downsampler_params = config.get("downsampler_params", {})
    downsampler = downsampler_class(**downsampler_params)
    model = model_class(downsampler=downsampler, **params)

    # Main network only (exclude downsampler for comparability)
    if hasattr(model, "conv_net"):
        # Arch_02, Arch_04
        main_params = sum(p.numel() for p in model.conv_net.parameters())
    else:
        total = sum(p.numel() for p in model.parameters())
        down_params = sum(p.numel() for p in model.downsampler.parameters())
        main_params = total - down_params

    size_mb = main_params * 4 / 1e6
    results.append((arch_name, main_params, size_mb))
    print(f"{arch_name}: {main_params:,} params, {size_mb:.2f} MB")

print("\n--- Table values (Parameters, Size MB) ---")
for arch_name, n, mb in results:
    print(f"{arch_name}: {n}, {mb:.2f}")
