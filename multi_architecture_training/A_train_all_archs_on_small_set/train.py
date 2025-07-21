# A_train_all_archs_on_small_set/train.py
"""
Train multiple architectures on the small dataset with per-model hyperparameters.
"""

import torch
from config.paths_config import DATALOADERS_DIR
from multi_architecture_training.training_utils.weights_init import weights_init
from multi_architecture_training.training_utils.train_model import train_model
from multi_architecture_training.A_train_all_archs_on_small_set.architecture_configs import architectures

def run_train_all_on_small():
    """
    Train all architectures on the small dataset.
    Saves each model and its loss curves.
    """
    # Load dataloaders
    train_loader = torch.load(DATALOADERS_DIR / 'small_train_loader.pt')
    val_loader = torch.load(DATALOADERS_DIR / 'small_val_loader.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for arch_name, config in architectures.items():
        # Initialize downsampler and model with their respective parameters
        downsampler = config["downsampler_class"](**config["downsampler_params"]).to(device)
        model = config["model_class"](downsampler=downsampler, **config["params"]).to(device)

        # Apply weight initialization
        model.apply(lambda m: weights_init(m, weight_init='xavier'))
        downsampler.apply(lambda m: weights_init(m, weight_init='xavier'))

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        # Train model and save losses
        train_model(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10, #config["epochs"],
            arch_name=arch_name,
            device=device,
            stage='A'
        )
