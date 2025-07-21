# training_utils/train_model.py

import torch
import warnings
from tqdm import tqdm
from config.paths_config import get_model_path, get_losses_path

# Suppress CUDNN warnings
warnings.filterwarnings("ignore", message=".*cudnnException.*")

def train_model(model, optimizer, train_loader, val_loader, num_epochs, arch_name, device, stage):
    """
    Train a model and save results
    
    Args:
        model: PyTorch model to train
        optimizer: PyTorch optimizer
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        arch_name: Architecture name for saving
        device: Device to train on
        stage: Stage of training ('A', 'B' or 'C')
    """
    criterion = torch.nn.L1Loss()
    
    train_losses = []
    val_losses = []

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc=f"Training {arch_name}", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        for terrain, input_data, label in train_loader:
            terrain, input_data, label = terrain.to(device), input_data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(terrain, input_data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for terrain, input_data, label in val_loader:
                terrain, input_data, label = terrain.to(device), input_data.to(device), label.to(device)
                output = model(terrain, input_data)
                loss = criterion(output, label)
                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Update progress bar with current losses
        epoch_pbar.set_postfix({'Train Loss': f'{avg_train_loss:.6f}','Val Loss': f'{avg_val_loss:.6f}'})

    # Save trained model using config paths
    model_path = get_model_path(stage, arch_name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # Save loss curves using config paths
    losses_path = get_losses_path(stage, arch_name)
    losses_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'train_losses': train_losses, 'val_losses': val_losses}, losses_path)