import torch 

def calculate_dummy_mean_loss(netG, val_loader, device):
    netG.eval()
    all_dummy_diffs = []
    
    with torch.no_grad():
        for terrains, data, labels in val_loader:
            terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)
                        
            # Dummy model prediction (outputs zeros)
            y_dummy = torch.zeros_like(labels).to(device)
            dummy_diffs = torch.abs(y_dummy - labels)
            all_dummy_diffs.append(dummy_diffs)
    
    # Concatenate all absolute differences
    all_dummy_diffs = torch.cat(all_dummy_diffs)
    avg_dummy_val_loss = all_dummy_diffs.mean().item()

    
    return avg_dummy_val_loss













if False:
    def calculate_val_loss(netG, val_loader, loss, device):
        netG.eval()
        total_val_loss = 0.0
        total_dummy_loss = 0.0
        num_batches_val = len(val_loader)
        
        with torch.no_grad():
            for terrains, data, labels in val_loader:
                terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)
                
                # Model prediction
                y_fake = netG(terrains, data)
                val_loss = loss(y_fake, labels)
                total_val_loss += val_loss.item()
                
                # Dummy model prediction (outputs zeros)
                y_dummy = torch.zeros_like(labels).to(device)
                dummy_loss = loss(y_dummy, labels)
                total_dummy_loss += dummy_loss.item()
        
        avg_netG_val_loss = total_val_loss / num_batches_val
        avg_dummy_val_loss = total_dummy_loss / num_batches_val
        
        return avg_netG_val_loss, avg_dummy_val_loss
