# /home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Functions/gradients.py

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.lines import Line2D

class GradientTracker:
    def __init__(self):
        self.epoch_gradients = defaultdict(lambda: {'mean': [], 'max': []})

    def track_gradients(self, named_parameters, epoch):
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n:
                self.epoch_gradients[n]['mean'].append(p.grad.abs().mean().item())
                self.epoch_gradients[n]['max'].append(p.grad.abs().max().item())

    def plot_gradients(self):
        layers = list(self.epoch_gradients.keys())
        epochs = len(self.epoch_gradients[layers[0]]['mean'])  # assuming all layers have the same number of epochs

        for layer in layers:
            mean_grads = self.epoch_gradients[layer]['mean']
            max_grads = self.epoch_gradients[layer]['max']
            plt.plot(range(epochs), mean_grads, label=f"{layer} mean", alpha=0.6)
            plt.plot(range(epochs), max_grads, label=f"{layer} max", alpha=0.6)
        
        plt.xlabel("Epochs")
        plt.ylabel("Gradient value")
        plt.title("Gradient flow across epochs")
        plt.grid(True)
        plt.legend()
        plt.show()
