# C_train_best_four_on_big_set/train.py
"""
Train the best four architectures on the big dataset.
"""

import torch
from config.paths_config import DATALOADERS_DIR
from multi_architecture_training.training_utils.train_model import train_model
from multi_architecture_training.C_train_best_four_on_big_set.architecture_configs import architectures


def run_train_best_four_on_big():
    """
    Train the best four architectures on the big dataset.
    Saves each model and its loss curves.
    """
    # Load dataloaders
    train_loader = torch.load(DATALOADERS_DIR / 'big_train_loader.pt')
    val_loader = torch.load(DATALOADERS_DIR / 'big_val_loader.pt')

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
            stage="C"
        )
