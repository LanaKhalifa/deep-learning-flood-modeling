#%% test_trained_model_on_test_set
import json
import torch 
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from plot_samples_diff import plot_samples_diff
# Test the trained model on the test set and calculate the mean error and standard deviation
def test_trained_model_on_test_set(netG, val_loader, num_samples, Architecture_num, trial_num, num_epochs):
    netG.eval()
    with torch.no_grad():
        terrains, data, labels = next(iter(val_loader))
        data = data.to(device)
        labels = labels.to(device)
        y_fake = netG(terrains, data)
        
        plot_samples_diff(epoch, 'val', y_fake[:, 0, :, :], labels[:,0,:,:], plot_dir_val)
        
        plot_dir_test = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/{Architecture_num}/{trial_num}/samples/val/'
        os.makedirs(plot_dir_test, exist_ok=True)
        plot_samples_diff(epoch=num_epochs, which='val', prediction_diffs=y_fake[:, 0, :, :], true_diffs=lbls[:, 0, :, :], base_dir=plot_dir_test, num_samples=num_samples)


            
    netG.eval()
    with torch.no_grad():
        y_fake = netG(val_terrains, val_data)                       
        # Calculate the error
        errors = torch.abs(y_fake - val_labels)
        mean_error = errors.mean()
        std_error = errors.std()
        
        # Save the mean error and standard deviation to the test folder
        error_stats = {"mean_error": mean_error,
                       "std_error": std_error}
        
        error_stats_path = os.path.join(plot_dir_test, 'error_stats.json')
        
        with open(error_stats_path, 'w') as f:
            json.dump(error_stats, f)
        
        print(f"Mean Error: {mean_error}")
        print(f"Standard Deviation of Error: {std_error}")
                        
