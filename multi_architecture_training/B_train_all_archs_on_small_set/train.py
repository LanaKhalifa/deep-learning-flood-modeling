# B_train_all_archs_on_small_set/train.py
"""
Train multiple architectures on the small dataset with per-model hyperparameters.
"""

import torch
from config.paths_config import DATALOADERS_DIR
from multi_architecture_training.training_utils.train_model import train_model
from multi_architecture_training.B_train_all_archs_on_small_set.architecture_configs import architectures

def run_train_all_on_small():
    """
    Train all architectures on the small dataset.
    Saves each model and its loss curves.
    """
    # Load dataloaders
    train_loader = torch.load(DATALOADERS_DIR / 'small_train_loader.pt')
    val_loader = torch.load(DATALOADERS_DIR / 'small_val_loader.pt')

    for arch_name, config in architectures.items():
        # Initialize downsampler and model with their respective parameters
        downsampler = config["downsampler_class"](**config["downsampler_params"])
        model = config["model_class"](downsampler=downsampler, **config["params"])

        # Train model and save losses
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            arch_name=arch_name,
            stage='B'
        )
