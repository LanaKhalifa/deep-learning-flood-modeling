# /home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Functions/gradient_visualization.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

class GradientVisualizer:
    def __init__(self):
        self.epoch_gradients = []

    def collect_gradients(self, named_parameters):
        '''Collects the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backward() as 
        "visualizer.collect_gradients(self.model.named_parameters())" to collect the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item())
                max_grads.append(p.grad.abs().max().item())
        
        self.epoch_gradients.append((layers, ave_grads, max_grads))

    def plot_gradients(self):
        '''Plots the collected gradients over epochs.'''
        num_epochs = len(self.epoch_gradients)
        layers = self.epoch_gradients[0][0]  # Layer names

        # Initialize lists to hold gradients over epochs
        mean_grads_over_epochs = {layer: [] for layer in layers}
        max_grads_over_epochs = {layer: [] for layer in layers}

        # Populate the lists
        for epoch in range(num_epochs):
            _, ave_grads, max_grads = self.epoch_gradients[epoch]
            for i, layer in enumerate(layers):
                mean_grads_over_epochs[layer].append(ave_grads[i])
                max_grads_over_epochs[layer].append(max_grads[i])

        # Plot mean gradients over epochs
        plt.figure(figsize=(20, 10))
        for layer in layers:
            plt.plot(mean_grads_over_epochs[layer], label=f'{layer} mean')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Gradient')
        plt.title('Mean Gradients over Epochs')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        # Plot max gradients over epochs
        plt.figure(figsize=(20, 10))
        for layer in layers:
            plt.plot(max_grads_over_epochs[layer], label=f'{layer} max')
        plt.xlabel('Epochs')
        plt.ylabel('Max Gradient')
        plt.title('Max Gradients over Epochs')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

