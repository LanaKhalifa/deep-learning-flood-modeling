import matplotlib.pyplot as plt

def plot_samples(train_loader, samples_to_plot=10):
    """
    Plot samples from the train_loader.
    
    Parameters:
    - train_loader: DataLoader containing the training data.
    - samples_to_plot: Number of samples to plot.
    """
    for idx, (terrain, data, label) in enumerate(train_loader):
        if idx >= 1:
            break
        for i in range(data.shape[0]):  # Loop through batch
            if i >= samples_to_plot:
                break
            fig, axs = plt.subplots(1, 4, figsize=(28, 8))
            
            im0 = axs[0].imshow(terrain[i].squeeze(), cmap='terrain')
            axs[0].set_title('Terrain (m)', fontsize=35)
            cbar0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
            cbar0.ax.tick_params(labelsize=30)
            
            im1 = axs[1].imshow(data[i, 1].squeeze(), cmap='Blues')
            axs[1].set_title('Water Depth at n (m)', fontsize=35)
            cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
            cbar1.ax.tick_params(labelsize=30)
            
            im2 = axs[2].imshow(data[i, 0].squeeze(), cmap='Blues')
            axs[2].set_title('BC at n+1 (m)', fontsize=35)
            cbar2 = fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(labelsize=20)
            
            im4 = axs[3].imshow(label[i].squeeze(), cmap='Blues')
            axs[3].set_title('Water Depth Difference (m)', fontsize=35)
            cbar4 = fig.colorbar(im4, ax=axs[3], fraction=0.046, pad=0.04)
            cbar4.ax.tick_params(labelsize=20)
            
            plt.tight_layout()
            plt.show()

# Assuming train_loader is defined and available
# plot_samples(train_loader)



if False: 
    def plot_samples(train_loader, samples_to_plot=10):
        """
        Plot samples from the train_loader.
        
        Parameters:
        - train_loader: DataLoader containing the training data.
        - samples_to_plot: Number of samples to plot.
        """
        for idx, (terrain, data, label) in enumerate(train_loader):
            if idx >= 1:
                break
            for i in range(data.shape[0]):  # Loop through batch
                if i >= samples_to_plot:
                    break
                fig, axs = plt.subplots(1, 4, figsize=(20, 5))
                
                im0 = axs[0].imshow(terrain[i].squeeze(), cmap='viridis')
                axs[0].set_title('Terrain')
                fig.colorbar(im0, ax=axs[0])
                
                im1 = axs[1].imshow(data[i, 1].squeeze(), cmap='viridis')
                axs[1].set_title('Depth')
                fig.colorbar(im1, ax=axs[1])
                
                im2 = axs[2].imshow(data[i, 0].squeeze(), cmap='viridis')
                axs[2].set_title('BC')
                fig.colorbar(im2, ax=axs[2])
                
                im4 = axs[3].imshow(label[i].squeeze(), cmap='viridis')
                axs[3].set_title('Depth Difference')
                fig.colorbar(im4, ax=axs[3])
                
                plt.show()
