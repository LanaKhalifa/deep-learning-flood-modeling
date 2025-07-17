# multi_architecture_training/D_boxplots_RAE_all_sets/plot_boxplots.py

import torch
import os
from multi_architecture_training.D_boxplots_RAE_all_sets.rae_utils import calculate_rae_quartiles
from multi_architecture_training.D_boxplots_RAE_all_sets.config_sets_models import MODEL_CONFIGS
from multi_architecture_training.models.terrain_downsampler_k11s10 import TerrainDownsampleK11S10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shared downsampler
downsampler = TerrainDownsampleK11S10(c_start=1, c1=16, c2=8, c_end=1, act='leakyrelu').to(device)

# Output directory
output_dir = "multi_architecture_training/D_boxplots_RAE_all_sets/boxplots"
os.makedirs(output_dir, exist_ok=True)

for config_name, cfg in MODEL_CONFIGS.items():
    print(f"→ Processing {config_name}")

    # Load model
    model = cfg['arch_class'](downsampler=downsampler, **cfg['params']).to(device)
    model.load_state_dict(torch.load(cfg['model_path'], map_location=device))

    # Load loader
    loader = torch.load(cfg['loader_path'])

    # Run and save plot
    save_path = os.path.join(output_dir, f"{config_name}.png")
    calculate_rae_quartiles(model, loader, device, save_path=save_path, label=cfg['box_label'])
