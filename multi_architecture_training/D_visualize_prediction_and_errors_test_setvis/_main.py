
### E_visualize_prediction_and_errors_test_set/vis_main.py

import torch
import os
from multi_architecture_training.C_train_best_three_on_big_set.architecture_configs import architectures
from E_visualize_prediction_and_errors_test_set.vis_plotter import visualize_on_loader

# Paths to model weights
MODEL_PATHS = {
    "Arch_04": "multi_architecture_training/C_train_best_three_on_big_set/saved_trained_models/Arch_04/model.pth",
    "Arch_05": "multi_architecture_training/C_train_best_three_on_big_set/saved_trained_models/Arch_05/model.pth",
    "Arch_07": "multi_architecture_training/C_train_best_three_on_big_set/saved_trained_models/Arch_07/model.pth"
}

# Test loaders to visualize
from config import ROOT_DATALOADERS
LOADER_FILES = {
    "big_test": os.path.join(ROOT_DATALOADERS, "big_test_loader.pt"),
    "prj_03_test": os.path.join(ROOT_DATALOADERS, "prj_03_test_loader.pt")
}


def run_all_visualizations(device='cuda'):
    for arch_name, config in architectures.items():
        print(f"🔹 Visualizing predictions for {arch_name}...")

        # Load model
        model_class = config['model_class']
        model_params = config['params']
        model = model_class(**model_params).to(device)
        model_path = MODEL_PATHS[arch_name]
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        for loader_name, loader_path in LOADER_FILES.items():
            print(f"   ▶ On loader: {loader_name}")
            loader = torch.load(loader_path)

            save_dir = os.path.join("E_visualize_prediction_and_errors_test_set/figures", arch_name, loader_name)
            os.makedirs(save_dir, exist_ok=True)

            visualize_on_loader(model, loader, device, save_dir)

        print(f"✅ Done with {arch_name}\n")


if __name__ == '__main__':
    run_all_visualizations()
