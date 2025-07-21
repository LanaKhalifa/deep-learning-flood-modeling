# Errors on val
netG.cpu()
netG.eval()
with torch.no_grad():
    y_fake = netG(val_terrains, val_data)                       
    errors = torch.abs(y_fake - val_labels)
    mean_error = errors.mean().item()  # Convert to Python float
    std_error = errors.std().item()    # Convert to Python float
    
    # Save the mean error and standard deviation to the test folder
    error_stats = {"mean_error": mean_error,
                   "std_error": std_error}
    
    errors_val_dir = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/{Architecture_num}/{trial_num}/errors/val/'
    os.makedirs(errors_val_dir, exist_ok=True)
    error_stats_path = os.path.join(errors_val_dir, 'error_stats.json')
    with open(error_stats_path, 'w') as f:
        json.dump(error_stats, f)
    
# Errors on train
netG.eval()
with torch.no_grad():
    y_fake = netG(train_terrains, train_data)                       
    errors = torch.abs(y_fake - train_labels)
    mean_error = errors.mean().item()  # Convert to Python float
    std_error = errors.std().item()    # Convert to Python float
    
    # Save the mean error and standard deviation to the test folder
    error_stats = {"mean_error": mean_error,
                   "std_error": std_error}
    
    errors_train_dir = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/{Architecture_num}/{trial_num}/errors/train/'
    os.makedirs(errors_train_dir, exist_ok=True)
    error_stats_path = os.path.join(errors_train_dir, 'error_stats.json')
    with open(error_stats_path, 'w') as f:
        json.dump(error_stats, f)
    