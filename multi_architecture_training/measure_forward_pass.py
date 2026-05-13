#!/usr/bin/env python
"""Measure forward-pass time (ms per batch) for an architecture using a loaded batch.

Example: Arch_02 (Non-downsampling Convolutions)
  conda activate flood_env
  PYTHONPATH=. python multi_architecture_training/measure_forward_pass.py --arch Arch_02

Optional: --arch Arch_04, --num_warmup 5, --num_repeats 100.
"""
import argparse
import time
import torch
from config.paths_config import DATALOADERS_DIR
from config.model_configs import get_stage_B_configs


def measure_forward_ms(arch_name, num_warmup=10, num_repeats=100):
    """Load one batch, build model, run warmup then timed forwards; return mean ms per batch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = get_stage_B_configs()
    if arch_name not in configs:
        raise ValueError(f"Unknown arch: {arch_name}. Choose from {list(configs.keys())}")

    # Load dataloader and get one batch
    loader_path = DATALOADERS_DIR / "small_train_loader.pt"
    if not loader_path.exists():
        raise FileNotFoundError(
            f"Dataloader not found: {loader_path}. Run generate_dataloaders first."
        )
    train_loader = torch.load(loader_path, weights_only=False)
    batch = next(iter(train_loader))
    terrain, input_data, label = batch
    terrain = terrain.to(device)
    input_data = input_data.to(device)
    batch_size = terrain.shape[0]

    # Build model (Stage B config)
    config = configs[arch_name]
    downsampler = config["downsampler_class"](**config["downsampler_params"])
    model = config["model_class"](downsampler=downsampler, **config["params"])
    model = model.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(terrain, input_data)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed forward passes
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(num_repeats):
            _ = model(terrain, input_data)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    total_s = end - start
    mean_ms = (total_s / num_repeats) * 1000
    return mean_ms, batch_size, device.type


def main():
    parser = argparse.ArgumentParser(description="Measure forward-pass time (ms/batch)")
    parser.add_argument(
        "--arch",
        type=str,
        default="Arch_02",
        help="Architecture name (e.g. Arch_02, Arch_04)",
    )
    parser.add_argument("--num_warmup", type=int, default=10, help="Warmup forward passes")
    parser.add_argument("--num_repeats", type=int, default=100, help="Timed forward passes")
    args = parser.parse_args()

    mean_ms, batch_size, device = measure_forward_ms(
        args.arch, num_warmup=args.num_warmup, num_repeats=args.num_repeats
    )
    print(f"{args.arch}: {mean_ms:.3f} ms/batch (batch_size={batch_size}, device={device})")


if __name__ == "__main__":
    main()
