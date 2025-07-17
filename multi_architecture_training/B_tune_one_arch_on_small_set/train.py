# multi_architecture_training/B_tune_one_arch_on_small_set/train.py

import torch
import os
from pathlib import Path

from config import DATALOADERS_ROOT
from multi_architecture_training.training_utils.weights_init import weights_init
from multi_architecture_training.training_utils.train_model import train_model
from multi_architecture_training.B_tune_one_arch_on_small_set.arch_04_tuned_config import final_config

CURRENT_DIR = Path(__file__).parent

def run_train_tuned_arch_04_on_small():
    train_loader = torch.load(os.path.join(DATALOADERS_ROOT, 'small_train_loader.pt'))
    val_loader = torch.load(os.path.join(DATALOADERS_ROOT, 'small_val_loader.pt'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_name = final_config["arch_name"]

    print(f"\n🔧 Training tuned model: {arch_name}")

    downsampler = final_config["downsampler_class"](**final_config["downsampler_params"]).to(device)
    model = final_config["model_class"](downsampler=downsampler, **final_config["params"]).to(device)

    model.apply(lambda m: weights_init(m, weight_init='xavier'))
    downsampler.apply(lambda m: weights_init(m, weight_init='xavier'))

    optimizer = torch.optim.Adam(model.parameters(), lr=final_config["lr"])

    train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=final_config["epochs"],
        arch_name=arch_name,
        device=device,
        save_root_dir=CURRENT_DIR
    )
