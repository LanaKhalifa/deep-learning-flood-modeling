import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.lines import Line2D
import os
from mpl_toolkits.mplot3d import Axes3D


class GradientTracker:
    def __init__(self):
        self.epoch_gradients = defaultdict(lambda: {'mean': [], 'max': []})

    def track_gradients(self, named_parameters):
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n:
                self.epoch_gradients[n]['mean'].append(p.grad.abs().mean().item())
                self.epoch_gradients[n]['max'].append(p.grad.abs().max().item())

    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    def plot_gradients(self, gradients_dir, window_size=5):
        os.makedirs(gradients_dir, exist_ok=True)
        layers = list(self.epoch_gradients.keys())
        epochs = len(self.epoch_gradients[layers[0]]['mean'])  # assuming all layers have the same number of epochs
    
        # Create 3D plot
        fig = plt.figure(figsize=(20, 18))
        ax = fig.add_subplot(111, projection='3d')
        
        for layer_idx, layer in enumerate(layers):
            mean_grads = self.epoch_gradients[layer]['mean']
            iterations = range(len(mean_grads))
            layer_indices = np.full_like(iterations, layer_idx)  # Create a constant z-axis value for each layer
        
            # Plot original mean gradients
            ax.plot(iterations, layer_indices, mean_grads, label=f"{layer} mean", alpha=0.6)
        
        # Set labels with increased font size
        ax.set_xlabel('Iterations', fontsize=15)
        ax.set_ylabel('Layer Indices', fontsize=15)
        ax.set_zlabel('Mean Gradients', fontsize=15)
        
        # Set title with increased font size
        ax.set_title('3D Plot of Mean Gradient Flow across Epochs and Layers', fontsize=20)
        
        # Set tick parameters with increased font size
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        
        # Set legend with increased font size
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15)
        
        # Adjust layout to make room for the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        plt.show()
        
        # Plot mean gradients for each layer in separate plots
        for layer in layers:
            plt.figure(figsize=(10, 6))
            mean_grads = self.epoch_gradients[layer]['mean']
            plt.plot(range(epochs), mean_grads, label=f"{layer} mean", alpha=0.6)
            
            # Compute and plot smoothed mean gradients
            if len(mean_grads) >= window_size:
                smoothed_grads = self.moving_average(mean_grads, window_size)
                plt.plot(range(window_size - 1, epochs), smoothed_grads, label=f"{layer} mean (smoothed)", alpha=0.8)
                
            plt.xlabel("Iterations")
            plt.ylabel("Gradient value")
            plt.title(f"Mean Gradient Flow for {layer} across Epochs")
            plt.grid(True)
            plt.legend()
            plt_path = os.path.join(gradients_dir, f"mean_gradients_{layer}.png")
            plt.savefig(plt_path, bbox_inches='tight')
            plt.show()
            plt.close()

if False:
    # /home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Functions/GradientTracker.py
    
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    from matplotlib.lines import Line2D
    import os
    
    class GradientTracker:
        def __init__(self):
            self.epoch_gradients = defaultdict(lambda: {'mean': [], 'max': []})
    
        def track_gradients(self, named_parameters):
            for n, p in named_parameters:
                if p.requires_grad and "bias" not in n:
                    self.epoch_gradients[n]['mean'].append(p.grad.abs().mean().item())
                    self.epoch_gradients[n]['max'].append(p.grad.abs().max().item())
    
        def plot_gradients(self, gradients_dir):
            os.makedirs(gradients_dir, exist_ok=True)
            layers = list(self.epoch_gradients.keys())
            epochs = len(self.epoch_gradients[layers[0]]['mean'])  # assuming all layers have the same number of epochs
        
            # Plot mean gradients for all layers in a single plot
            plt.figure(figsize=(10, 6))
            for layer in layers:
                mean_grads = self.epoch_gradients[layer]['mean']
                plt.plot(range(epochs), mean_grads, label=f"{layer} mean", alpha=0.6)
            
            plt.xlabel("Iterations")
            plt.ylabel("Gradient value")
            plt.title("Mean Gradient Flow across Epochs")
            plt.grid(True)
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
            plt_path = os.path.join(gradients_dir, "mean_gradients_all_layers.png")
            plt.savefig(plt_path, bbox_inches='tight')
            plt.show()
            plt.close()
        
            # Plot mean gradients for each layer in separate plots
            for layer in layers:
                plt.figure(figsize=(10, 6))
                mean_grads = self.epoch_gradients[layer]['mean']
                plt.plot(range(epochs), mean_grads, label=f"{layer} mean", alpha=0.6)
                plt.xlabel("Iterations")
                plt.ylabel("Gradient value")
                plt.title(f"Mean Gradient Flow for {layer} across Epochs")
                plt.grid(True)
                plt.legend()
                plt_path = os.path.join(gradients_dir, f"mean_gradients_{layer}.png")
                plt.savefig(plt_path, bbox_inches='tight')
                plt.show()
                plt.close()
    
    
