import torch
import os
import numpy as np

def analyze_all_losses():
    """Load and analyze losses for all architectures."""
    base_path = "multi_architecture_training/B_train_all_archs_on_small_set/saved_losses"
    
    # Get all architecture directories
    arch_dirs = [d for d in os.listdir(base_path) if d.startswith('Arch_')]
    arch_dirs.sort(key=lambda x: int(x.split('_')[1]))  # Sort by architecture number
    
    print("=== COMPREHENSIVE LOSS ANALYSIS ===")
    print("=" * 50)
    
    # Store results for comparison
    results = {}
    
    for arch_name in arch_dirs:
        loss_file = os.path.join(base_path, arch_name, "losses.pt")
        
        if not os.path.exists(loss_file):
            print(f"Loss file not found for {arch_name}")
            continue
        
        try:
            # Load the losses
            losses = torch.load(loss_file)
            
            print(f"\n--- {arch_name} ---")
            
            if isinstance(losses, dict) and 'train_losses' in losses and 'val_losses' in losses:
                train_losses = losses['train_losses']
                val_losses = losses['val_losses']
                
                if isinstance(train_losses, list):
                    train_losses = torch.tensor(train_losses)
                if isinstance(val_losses, list):
                    val_losses = torch.tensor(val_losses)
                
                train_min = torch.min(train_losses).item()
                train_max = torch.max(train_losses).item()
                train_mean = torch.mean(train_losses).item()
                train_std = torch.std(train_losses).item()
                
                val_min = torch.min(val_losses).item()
                val_max = torch.max(val_losses).item()
                val_mean = torch.mean(val_losses).item()
                val_std = torch.std(val_losses).item()
                
                print(f"Training Loss:")
                print(f"  Min: {train_min:.6f}")
                print(f"  Max: {train_max:.6f}")
                print(f"  Mean: {train_mean:.6f}")
                print(f"  Std: {train_std:.6f}")
                print(f"  Epochs: {len(train_losses)}")
                
                print(f"Validation Loss:")
                print(f"  Min: {val_min:.6f}")
                print(f"  Max: {val_max:.6f}")
                print(f"  Mean: {val_mean:.6f}")
                print(f"  Std: {val_std:.6f}")
                print(f"  Epochs: {len(val_losses)}")
                
                # Store results for comparison
                results[arch_name] = {
                    'train_min': train_min,
                    'train_mean': train_mean,
                    'val_min': val_min,
                    'val_mean': val_mean,
                    'train_std': train_std,
                    'val_std': val_std
                }
                
            else:
                print(f"Unexpected data structure for {arch_name}")
                print(f"Type: {type(losses)}")
                if isinstance(losses, dict):
                    print(f"Keys: {list(losses.keys())}")
                
        except Exception as e:
            print(f"Error loading losses for {arch_name}: {e}")
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("SUMMARY COMPARISON")
    print("=" * 50)
    
    if results:
        print(f"{'Arch':<8} {'Train Min':<10} {'Train Mean':<11} {'Val Min':<9} {'Val Mean':<10} {'Train Std':<10} {'Val Std':<9}")
        print("-" * 70)
        
        for arch_name in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
            r = results[arch_name]
            print(f"{arch_name:<8} {r['train_min']:<10.6f} {r['train_mean']:<11.6f} {r['val_min']:<9.6f} {r['val_mean']:<10.6f} {r['train_std']:<10.6f} {r['val_std']:<9.6f}")
        
        # Find best performers
        best_train_min = min(results.items(), key=lambda x: x[1]['train_min'])
        best_val_min = min(results.items(), key=lambda x: x[1]['val_min'])
        best_train_mean = min(results.items(), key=lambda x: x[1]['train_mean'])
        best_val_mean = min(results.items(), key=lambda x: x[1]['val_mean'])
        
        print(f"\nBEST PERFORMERS:")
        print(f"Lowest Training Loss: {best_train_min[0]} ({best_train_min[1]['train_min']:.6f})")
        print(f"Lowest Validation Loss: {best_val_min[0]} ({best_val_min[1]['val_min']:.6f})")
        print(f"Best Average Training Loss: {best_train_mean[0]} ({best_train_mean[1]['train_mean']:.6f})")
        print(f"Best Average Validation Loss: {best_val_mean[0]} ({best_val_mean[1]['val_mean']:.6f})")

if __name__ == "__main__":
    analyze_all_losses()

