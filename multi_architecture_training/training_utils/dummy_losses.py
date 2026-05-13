"""
Dummy loss calculation utilities.
Functions to calculate baseline losses when predicting all zeros.
"""

import torch
from pathlib import Path
from config.paths_config import DATALOADERS_DIR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_dummy_mean_loss(data_loader):
    """Calculate L1 loss when predicting all zeros (same as training loop)"""
    criterion = torch.nn.L1Loss()
    running_loss = 0.0
    
    with torch.no_grad():
        for terrains, data, labels in data_loader:
            terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)
            y_dummy = torch.zeros_like(labels).to(device)
            loss = criterion(y_dummy, labels)
            running_loss += loss.item()
    
    avg_dummy_loss = running_loss / len(data_loader)
    return avg_dummy_loss

def calculate_and_save_dummy_losses():
    """Calculate and save dummy losses for both small and big validation datasets"""
    small_val_loader = torch.load(DATALOADERS_DIR / 'small_val_loader.pt')
    big_val_loader = torch.load(DATALOADERS_DIR / 'big_val_loader.pt')

    dummy_small_val_loss = calculate_dummy_mean_loss(small_val_loader)
    dummy_big_val_loss = calculate_dummy_mean_loss(big_val_loader)
    
    training_utils_dir = Path(__file__).parent

    dummy_loss_small_val_path = training_utils_dir / "dummy_small_val_loss.pt"
    dummy_loss_big_val_path = training_utils_dir / "dummy_big_val_loss.pt"

    torch.save(dummy_small_val_loss, dummy_loss_small_val_path)
    torch.save(dummy_big_val_loss, dummy_loss_big_val_path)