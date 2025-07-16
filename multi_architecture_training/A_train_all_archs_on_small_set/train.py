# A_train_all_archs_on_small_set/train.py
"""
Train multiple architectures on the small dataset with per-model hyperparameters.
"""

import torch
import os

def train_all_architectures():
    from config import DATALOADERS_ROOT
    from training_utils.weights_init import weights_init
    from training_utils.train_model import train_model
    from A_train_all_archs_on_small_set.architecture_configs import architectures

    # Load dataloaders
    train_loader = torch.load(os.path.join(DATALOADERS_ROOT, 'small_train_val.pt'))
    test_loader = torch.load(os.path.join(DATALOADERS_ROOT, 'small_test.pt'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for arch_name, config in architectures.items():
        print(f"\nTraining {arch_name}...")

        downsampler = config["downsampler_class"](**config["downsampler_params"]).to(device)
        model = config["model_class"](downsampler=downsampler, **config["params"]).to(device)

        model.apply(lambda m: weights_init(m, weight_init='xavier'))
        downsampler.apply(lambda m: weights_init(m, weight_init='xavier'))

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        train_model(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=config["epochs"],
            arch_name=arch_name,
            device=device
        )

# Allow this file to be run directly as well
if __name__ == "__main__":
    train_all_architectures()
