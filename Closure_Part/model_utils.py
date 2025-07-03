
"""
model_utils.py

Utilities for loading the deep learning model and running inference.
"""

import torch
import numpy as np
import copy
from config import MODEL_PATH, CELLS_IN_PATCH
from TerrainDownsample_k11s1p0 import TerrainDownsample_k11s1p0
from arch_05 import arch_05

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_trained_model():
    """
    Loads the trained PyTorch model with shared terrain downsampler.

    Returns:
        model (torch.nn.Module): Loaded model in eval mode
    """
    down_c_start = 1
    down_c1 = 10
    down_c2 = 20
    down_c_end = 1

    shared_downsampler = TerrainDownsample_k11s1p0(down_c_start, down_c_end, down_c1, down_c2).to(device)
    model = arch_05(shared_downsampler).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model

def zero_internal(matrix):
    """
    Zeros out the internal values of the matrix, leaving a 2-cell border untouched.

    Args:
        matrix (np.ndarray): 2D input

    Returns:
        np.ndarray: Modified matrix
    """
    out = copy.deepcopy(matrix)
    out[2:-2, 2:-2] = 0
    return out

def fill_internal_with(matrix, filler):
    """
    Fills internal values of matrix with values from another matrix (for BC initialization).

    Args:
        matrix (np.ndarray): Matrix to modify
        filler (np.ndarray): Matrix to copy from

    Returns:
        np.ndarray: Modified matrix
    """
    out = copy.deepcopy(matrix)
    out[2:-2, 2:-2] = filler[2:-2, 2:-2]
    return out

def prepare_input_tensor(patches_depth, patches_next_depth):
    """
    Prepares model input tensor of shape (N, 2, H, W)

    Args:
        patches_depth (np.ndarray): Current depths (N, H, W)
        patches_next_depth (np.ndarray): Next depths with BCs only (N, H, W)

    Returns:
        torch.Tensor: Input tensor for model
    """
    bc_only = np.array([zero_internal(p) for p in patches_next_depth])
    input_data = np.stack([bc_only, patches_depth], axis=1)
    return torch.tensor(input_data, dtype=torch.double).to(device)

def prepare_tiff_tensor(tiff_patches):
    """
    Prepares TIFF terrain patches for model input.

    Args:
        tiff_patches (np.ndarray): Shape (N, H, W)

    Returns:
        torch.Tensor: Shape (N, 1, H, W)
    """
    return torch.tensor(np.expand_dims(tiff_patches, axis=1), dtype=torch.double).to(device)

def run_model(model, terrain_tensor, depth_tensor):
    """
    Runs the model on given input and returns predicted delta depths.

    Args:
        model: Trained model
        terrain_tensor: Input terrain tensor
        depth_tensor: Input depth tensor

    Returns:
        np.ndarray: Predicted delta depths (N, H, W)
    """
    with torch.no_grad():
        output = model(terrain_tensor, depth_tensor)
        return output.squeeze(1).cpu().numpy()
