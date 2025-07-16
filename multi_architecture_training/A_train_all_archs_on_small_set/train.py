# A_train_all_archs_on_small_set/train.py
"""
Train multiple architectures on the small dataset with per-model hyperparameters.
"""

import torch
import os
from config import DATALOADERS_ROOT
from training_utils.weights_init import weights_init
from training_utils.train_model import train_model
from A_train_all_archs_on_small_set.architecture_configs import architectures
from pathlib import Path
CURRENT_DIR = Path(__file__).parent


def run_train_all_on_small():
    """
    Train all architectures on the small dataset.
    Saves each model and its loss curves.
    """
    # Load dataloaders
    train_loader = torch.load(os.path.join(DATALOADERS_ROOT, 'small_train_val.pt'))
    test_loader = torch.load(os.path.join(DATALOADERS_ROOT, 'small_test.pt'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for arch_name, config in architectures.items():
        print(f"\n🔧 Training {arch_name}...")

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
            test_loader=test_loader,
            num_epochs=config["epochs"],
            arch_name=arch_name,
            device=device,
            save_root_dir=CURRENT_DIR
        )
