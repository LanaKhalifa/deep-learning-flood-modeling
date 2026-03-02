import torch
import os

def analyze_losses(arch_name):
    """Load and analyze losses for a given architecture."""
    loss_file = f"multi_architecture_training/B_train_all_archs_on_small_set/saved_losses/{arch_name}/losses.pt"
    
    if not os.path.exists(loss_file):
        print(f"Loss file not found for {arch_name}")
        return
    
    try:
        # Load the losses
        losses = torch.load(loss_file)
        
        print(f"\n=== {arch_name} Analysis ===")
        print(f"Loss file: {loss_file}")
        
        # Check the type and structure of the loaded data
        print(f"Type of loaded data: {type(losses)}")
        
        if isinstance(losses, dict):
            print(f"Keys in losses dict: {list(losses.keys())}")
            for key, value in losses.items():
                if isinstance(value, (list, torch.Tensor)):
                    if isinstance(value, list):
                        value = torch.tensor(value)
                    min_loss = torch.min(value).item()
                    max_loss = torch.max(value).item()
                    mean_loss = torch.mean(value).item()
                    print(f"  {key}:")
                    print(f"    Min loss: {min_loss:.6f}")
                    print(f"    Max loss: {max_loss:.6f}")
                    print(f"    Mean loss: {mean_loss:.6f}")
                    print(f"    Number of epochs: {len(value)}")
                else:
                    print(f"  {key}: {value}")
        
        elif isinstance(losses, (list, torch.Tensor)):
            if isinstance(losses, list):
                losses = torch.tensor(losses)
            min_loss = torch.min(losses).item()
            max_loss = torch.max(losses).item()
            mean_loss = torch.mean(losses).item()
            print(f"Min loss: {min_loss:.6f}")
            print(f"Max loss: {max_loss:.6f}")
            print(f"Mean loss: {mean_loss:.6f}")
            print(f"Number of epochs: {len(losses)}")
        
        else:
            print(f"Unexpected data type: {type(losses)}")
            print(f"Data: {losses}")
            
    except Exception as e:
        print(f"Error loading losses for {arch_name}: {e}")

# Analyze both architectures
analyze_losses("Arch_04")
analyze_losses("Arch_05")

