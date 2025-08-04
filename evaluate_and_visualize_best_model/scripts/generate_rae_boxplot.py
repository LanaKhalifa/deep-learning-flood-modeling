import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config.paths_config import RAE_BOXPLOTS_DIR, DATALOADERS_DIR

def clean_rae_data(rae_values):
    # Only remove infinite values
    return rae_values[np.isfinite(rae_values)]

def calculate_rae_one_sample(prediction, ground_truth):
    diff = torch.abs(ground_truth - prediction)
    diff_sum = torch.sum(diff, dim=(1, 2, 3))
    dummy_diff = torch.abs(ground_truth)
    dummy_diff_sum = torch.sum(dummy_diff, dim=(1, 2, 3))
    rae = diff_sum / dummy_diff_sum
    return rae.item()

def calculate_and_save_all_raes(device, arch_name='arch_04'):
    from multi_architecture_training.models.non_downsampling_convolutions_attention import NonDownsamplingConvolutionsWithAttention
    from multi_architecture_training.models.terrain_downsampler_alternating import TerrainDownsampleAlternating
    import os
    datasets = [
        'big_train',
        'big_val',
        'big_test',
        'prj_03_train_val',
        'prj_03_test'
    ]
    downsampler = TerrainDownsampleAlternating(c_start=1, c1=20, c2=40, c_end=1).to(device)
    model = NonDownsamplingConvolutionsWithAttention(
        downsampler=downsampler,
        arch_input_c=3,
        arch_num_layers=12,
        arch_num_c=32,
        arch_act="leakyrelu",
        arch_last_act="leakyrelu",
        arch_num_attentions=2
    ).to(device)
    model_path = 'multi_architecture_training/C_train_best_four_on_big_set/saved_trained_models/Arch_04/model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    rae_data = {}
    for dataset in datasets:
        dataloader_path = os.path.join(DATALOADERS_DIR, f'{dataset}_loader.pt')
        dataloader = torch.load(dataloader_path)
        all_raes = []
        with torch.no_grad():
            for terrains, data, labels in dataloader:
                terrains, data, labels = terrains.to(device), data.to(device), labels.to(device)
                predictions = model(terrains, data)
                diff = torch.abs(labels - predictions)
                diff_sum = torch.sum(diff, dim=(1, 2, 3))
                dummy_diff = torch.abs(labels)
                dummy_diff_sum = torch.sum(dummy_diff, dim=(1, 2, 3))
                batch_raes = diff_sum / dummy_diff_sum
                all_raes.extend(batch_raes.cpu().numpy())
        rae_values = np.array(all_raes)
        rae_clean = clean_rae_data(rae_values)
        output_dir = Path(RAE_BOXPLOTS_DIR) / arch_name
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f'{dataset}_rae_values.npy', rae_clean)
        rae_data[dataset] = rae_clean
    return rae_data

def plot_rae_boxplots_arch_04(device):
    arch_name='arch_04'
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Nimbus Roman']
    rae_data = calculate_and_save_all_raes(device, arch_name)
    available_datasets = list(rae_data.keys())
    boxplot_data = [rae_data[dataset] for dataset in available_datasets]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(boxplot_data, vert=False, patch_artist=True, showmeans=False, showfliers=False,
               medianprops=dict(color='blue', linewidth=2),
               boxprops=dict(color='black', linewidth=2, facecolor='lightgrey'),
               whiskerprops=dict(color='black', linewidth=2),
               capprops=dict(color='black', linewidth=2),
               flierprops=dict(marker='o', color='red', alpha=0.5),
               widths=0.5)
    ax.set_yticks([i + 1 for i in range(len(available_datasets))])
    ax.set_yticklabels(available_datasets, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Relative Absolute Error (RAE)', fontsize=16)
    ax.set_title(f'RAE Distribution Across Datasets - Non-downsampling Convolutions with Self-Attention', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.1)
    plt.tight_layout()
    output_dir = Path('evaluate_and_visualize_best_model/boxplots')
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f'{arch_name}_rae_boxplots.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved boxplot to: {save_path}")
    plt.show()
    return fig, ax

def main():
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_rae_boxplots_arch_04(device)

if __name__ == "__main__":
    main() 