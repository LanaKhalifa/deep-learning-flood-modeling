import pickle

import torch
from torch import nn
from tqdm import tqdm

from TerrainDownsample_alter import TerrainDownsample_alter

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
from load_list_from_file import load_list_from_file
from zero_internal import zero_internal
from SelfAttention import ConvSelfAttention
from plot_samples_diff_eval import plot_samples_diff_eval

# Settings
boundary_thickness = 5
batch_size = 200

torch.manual_seed(44)
np.random.seed(44)
torch.set_default_dtype(torch.float64)


def load_dataloader():
    print("Loading data...")
    # Load the data
    with open("data/depths_next_np_filtered.pkl", 'rb') as f:
        depths_next_numpy = pickle.load(f)
    with open("data/depths_np_filtered.pkl", 'rb') as f:
        depths_numpy = pickle.load(f)
    with open("data/terrains_np.pkl", 'rb') as f:
        terrains_numpy = pickle.load(f)
    print("Data loaded.")
    # Create entire dataset
    N = min(1000, int(terrains_numpy.shape[0]))

    terrains_expanded = np.expand_dims(copy.deepcopy(terrains_numpy), axis=1)
    terrains_tensor = torch.tensor(terrains_expanded).double()

    num_cells_in_patch = depths_numpy.shape[-1]
    dataset_numpy = np.zeros((N, 2, num_cells_in_patch, num_cells_in_patch))

    just_for_zeroing = copy.deepcopy(depths_next_numpy)
    depths_BC_numpy = zero_internal(just_for_zeroing, boundary_thickness)

    for i in tqdm(range(N)):
        stacked = np.stack((depths_BC_numpy[i], depths_numpy[i]))
        dataset_numpy[i] = stacked
    labels_numpy = np.expand_dims(depths_next_numpy - depths_numpy, axis=1)
    labels_tensor = torch.tensor(labels_numpy).double()

    dataset_tensor = torch.tensor(dataset_numpy).double()

    # Create dataloader

    val_terrains = terrains_tensor[:N]
    val_data = dataset_tensor[:N]
    val_labels = labels_tensor[:N]

    val_dataset = TensorDataset(val_terrains, val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_loader


class SimpleFlexibleCNN(nn.Module):
    def __init__(self, downsampler, num_layers=10, num_channels=1, input_channels=3):
        super(SimpleFlexibleCNN, self).__init__()
        self.nonlinearity = nn.LeakyReLU()
        self.downsampler = downsampler

        layers = []

        # layer 0
        layers.append(
            nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1))
        layers.append(self.nonlinearity)

        # layer 1, 2
        # Add half of the convolutional layers
        for _ in range(1, num_layers // 2):
            layers.append(
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1))
            layers.append(self.nonlinearity)

        # Add the ConvSelfAttention layer
        self.attention = ConvSelfAttention(num_channels)

        # layer 3, 4
        # Add the remaining convolutional layers
        for _ in range(num_layers // 2, num_layers - 1):
            layers.append(
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1))
            layers.append(self.nonlinearity)

        # layer 5
        layers.append(nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Tanh())

        self.conv_net = nn.Sequential(*layers)

    def forward(self, terrain, depths):
        terrain_out = self.downsampler(terrain)
        x = torch.cat([terrain_out, depths], dim=1)
        x = self.conv_net[:len(self.conv_net) // 2](x)
        x = self.attention(x)  # Apply the attention layer
        x = self.conv_net[len(self.conv_net) // 2:](x)
        return x


def load_model(torch_model_path: str):
    print("Loading model from", torch_model_path)
    terrain_down_sampler = TerrainDownsample_alter(c_start=1, c_end=1, c1=10, c2=20, act='leakyrelu')
    model = SimpleFlexibleCNN(terrain_down_sampler, num_layers=6, num_channels=32, input_channels=3)
    model.load_state_dict(torch.load(torch_model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def run():
    dataloader = load_dataloader()
    model = load_model("trained_model/arch_4_trained_model.pth")
    model.eval()

    sum = 0
    count = 0
    for i, (terrains, data, labels) in enumerate(dataloader):
        with torch.no_grad():
            model_output = model(terrains, data)
            batch_mean, RAE_batch_max, RAE_batch_median = plot_samples_diff_eval(i, "VAL", model_output, labels, terrains, data[:, 1, :, :], "plots", 50)
            print(f"Batch {i} Mean RAE:", batch_mean)
            print(f"Batch {i} Max RAE:", RAE_batch_max)
            print(f"Batch {i} Median RAE:", RAE_batch_median)
            sum += RAE_batch_median
            count += 1
    assert count > 0
    print("-"*50)
    print("Mean of Median?? RAE:", sum / count)


if __name__ == "__main__":
    run()
