import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # Import MaxNLocator

def plot_learning_curve(G_losses_train, G_losses_val, dummy_losses_train, dummy_losses_val, Architecture_num, trial_num, plt_path):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # Plot the training and validation losses on the left y-axis
    ax1.plot(G_losses_train, label='Training Loss', color='tab:blue', linestyle='solid')
    ax1.plot(G_losses_val, label='Validation Loss', color='tab:red', linestyle='solid')
    
    ax1.plot(dummy_losses_train, label='Training Dummy Loss', color='tab:blue', linestyle='--')
    ax1.plot(dummy_losses_val, label='Validation Dummy Loss', color='tab:red', linestyle='--')
    
    # Set labels with larger font sizes
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('L1 Loss', fontsize=20)
    
    # Increase tick label font sizes
    ax1.tick_params(axis='both', which='major', labelsize=20)
    
    # Ensure the x-axis shows integers (for epochs)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set the legend in the upper right corner
    ax1.legend(loc='upper right', fontsize=15)
    
    ax1.grid(True)
    
    # Set the title with a larger font size
    plt.title(f'{Architecture_num}, {trial_num}', fontsize=20)
    
    # Save and show the plot
    plt.savefig(plt_path)
    plt.show()
    plt.close()
