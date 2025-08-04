# training_utils/train_model.py

import torch
import warnings
from tqdm import tqdm
from config.paths_config import get_model_path, get_losses_path
from multi_architecture_training.training_utils.weights_init import weights_init

# Set default tensor type to double precision for all PyTorch operations
torch.set_default_dtype(torch.float64)

# Suppress CUDNN warnings
warnings.filterwarnings("ignore", message=".*cudnnException.*")

def train_model(model, train_loader, val_loader, arch_name, stage):
    """
    Train a model and save results
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        arch_name: Architecture name for saving
        stage: Stage of training ('A', 'B' or 'C')
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to device
    
    # Clear GPU cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Apply weight initialization with kaiming to all models
    model.apply(lambda m: weights_init(m, weight_init='kaiming'))
    
    criterion = torch.nn.L1Loss()
    num_epochs = 300  # Hardcoded number of epochs
    
    # Define optimizer with hardcoded learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        threshold=0.006, 
        cooldown=5,
        min_lr=1e-7
    )

    
    train_losses = []
    val_losses = []

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc=f"Training {arch_name}", unit="epoch", leave=False)
    
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

        # Step the scheduler with validation loss
        scheduler.step(avg_val_loss)
        
        # Get current learning rate for display
        current_lr = optimizer.param_groups[0]['lr']

        # Update progress bar with current losses and learning rate
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_train_loss:.6f}',
            'Val Loss': f'{avg_val_loss:.6f}',
            'LR': f'{current_lr:.8f}'
        })

    # Close progress bar properly
    epoch_pbar.close()
    
    # Clear GPU cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save trained model using config paths
    model_path = get_model_path(stage, arch_name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

    # Save loss curves using config paths
    losses_path = get_losses_path(stage, arch_name)
    losses_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'train_losses': train_losses, 'val_losses': val_losses}, losses_path)