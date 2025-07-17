# multi_architecture_training/D_boxplots_RAE_all_sets/rae_main.py

import torch
import os

from multi_architecture_training.C_train_best_three_on_big_set.architecture_configs import architectures
from multi_architecture_training.D_boxplots_RAE_all_sets.rae_plotter import calculate_rae_boxplot_all_sets

def run_all_rae_plots(device='cuda'):
    """
    Generates RAE boxplots for all three trained architectures from Stage C.
    Saves the figures in the D_boxplots_RAE_all_sets/figures directory.
    """
    arch_paths = {
        "Arch_04": "multi_architecture_training/C_train_best_three_on_big_set/saved_trained_models/Arch_04/model.pth",
        "Arch_05": "multi_architecture_training/C_train_best_three_on_big_set/saved_trained_models/Arch_05/model.pth",
        "Arch_07": "multi_architecture_training/C_train_best_three_on_big_set/saved_trained_models/Arch_07/model.pth"
    }

    save_dir = "multi_architecture_training/D_boxplots_RAE_all_sets/figures"
    os.makedirs(save_dir, exist_ok=True)

    for arch_name, config in architectures.items():
        print(f"🔹 Generating RAE boxplot for {arch_name}...")

        model_class = config['model_class']
        model_params = config['params']
        model = model_class(**model_params).to(device)

        model_path = arch_paths[arch_name]
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fig_path = os.path.join(save_dir, f"{arch_name}_rae_boxplot.png")
        calculate_rae_boxplot_all_sets(model, device, fig_path)

        print(f"✅ Saved RAE plot for {arch_name} → {fig_path}")
