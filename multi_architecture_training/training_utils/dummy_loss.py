import os
import torch

def calculate_dummy_mean_loss(data_loader, device):
    all_dummy_diffs = []
    with torch.no_grad():
        for terrains, data, labels in data_loader:
            terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)
            y_dummy = torch.zeros_like(labels).to(device)
            dummy_diffs = torch.abs(y_dummy - labels)
            all_dummy_diffs.append(dummy_diffs)
    return torch.cat(all_dummy_diffs).mean().item()

