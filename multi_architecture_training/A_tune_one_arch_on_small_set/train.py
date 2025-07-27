# A_tune_one_arch_on_small_set/train.py
"""
Train Arch_04 with tuned hyperparameters on the small dataloaders.
"""
import torch
from config.paths_config import DATALOADERS_DIR
from multi_architecture_training.training_utils.train_model import train_model
from multi_architecture_training.A_tune_one_arch_on_small_set.arch_04_tuned_config import final_config

def run_train_arch_04_tuned():
    """
    Train Arch_04 with tuned hyperparameters on the small dataset.
    Saves the model and its loss curves.
    """
    # Load dataloaders
    train_loader = torch.load(DATALOADERS_DIR / 'small_train_loader.pt')
    val_loader = torch.load(DATALOADERS_DIR / 'small_val_loader.pt')

    arch_name = "Arch_04"
     
    # Initialize downsampler and model with their respective parameters
    downsampler = final_config["downsampler_class"](**final_config["downsampler_params"])
    model = final_config["model_class"](downsampler=downsampler, **final_config["params"])

    # Train model and save losses
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        arch_name=arch_name,
        stage="A"
    )
