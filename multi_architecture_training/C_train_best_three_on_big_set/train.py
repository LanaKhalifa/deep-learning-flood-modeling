# multi_architecture_training/C_train_best_three_on_big_set/train.py

import torch
import os
from pathlib import Path

from config import DATALOADERS_ROOT
from multi_architecture_training.training_utils.weights_init import weights_init
from multi_architecture_training.training_utils.train_model import train_model
from multi_architecture_training.C_train_best_three_on_big_set.architecture_configs import architectures

CURRENT_DIR = Path(__file__).parent

def run_train_best_three_on_big():
    train_loader = torch.load(os.path.join(DATALOADERS_ROOT, 'big_train_loader.pt'))
    val_loader = torch.load(os.path.join(DATALOADERS_ROOT, 'big_val_loader.pt'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for arch_name, config in architectures.items():
        print(f"\n🔧 Training {arch_name} on big set...")

        downsampler = config["downsampler_class"](**config["downsampler_params"]).to(device)
        model = config["model_class"](downsampler=downsampler, **config["params"]).to(device)

        model.apply(lambda m: weights_init(m, weight_init='xavier'))
        downsampler.apply(lambda m: weights_init(m, weight_init='xavier'))

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        train_model(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config["epochs"],
            arch_name=arch_name,
            device=device,
            save_root_dir=CURRENT_DIR
        )
