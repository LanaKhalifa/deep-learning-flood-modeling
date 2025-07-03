"""
matrix_patcher.py

This module provides functions to extract and reconstruct 2D matrix patches.
It is used to divide large matrices (e.g., terrain, depth, or depth difference)
into overlapping patches and reassemble them after processing.

Author: [Your Name]
Created: [Date]
"""

import numpy as np

def extract_patches(matrix, patch_size, order_type):
    """
    Extract overlapping patches from a 2D matrix using a specified sweeping order.

    Parameters:
    - matrix (np.ndarray): The full domain 2D array (e.g., terrain, depth).
    - patch_size (int): Size of the square patch (e.g., 32 for 32x32).
    - order_type (str): One of {'A', 'B', 'C', 'D'} indicating sweeping order.

    Returns:
    - patches (list of np.ndarray): List of extracted patches.
    - num_patches_row (int): Number of patches along a row.
    - num_patches_col (int): Number of patches along a column.
    """
    stride = patch_size // 2
    offset_i = 0 if order_type in ['A', 'C'] else stride
    offset_j = 0 if order_type in ['A', 'B'] else stride

    n_rows, n_cols = matrix.shape
    patches = []

    for i in range(offset_i, n_rows - patch_size + 1, stride):
        for j in range(offset_j, n_cols - patch_size + 1, stride):
            patch = matrix[i:i+patch_size, j:j+patch_size]
            patches.append(patch)

    num_patches_row = (n_rows - patch_size - offset_i) // stride + 1
    num_patches_col = (n_cols - patch_size - offset_j) // stride + 1

    return patches, num_patches_row, num_patches_col

def rebuild_from_patches(patches, full_shape, patch_size, order_type):
    """
    Rebuild a full matrix from overlapping patches using averaging at overlaps.

    Parameters:
    - patches (list of np.ndarray): List of patches.
    - full_shape (tuple): Shape of the full output matrix (height, width).
    - patch_size (int): Size of the square patch.
    - order_type (str): One of {'A', 'B', 'C', 'D'} indicating sweeping order.

    Returns:
    - np.ndarray: The reconstructed full matrix.
    """
    stride = patch_size // 2
    offset_i = 0 if order_type in ['A', 'C'] else stride
    offset_j = 0 if order_type in ['A', 'B'] else stride

    out = np.zeros(full_shape)
    count = np.zeros(full_shape)
    n_rows, n_cols = full_shape

    patch_idx = 0
    for i in range(offset_i, n_rows - patch_size + 1, stride):
        for j in range(offset_j, n_cols - patch_size + 1, stride):
            out[i:i+patch_size, j:j+patch_size] += patches[patch_idx]
            count[i:i+patch_size, j:j+patch_size] += 1
            patch_idx += 1

    # Avoid division by zero
    count[count == 0] = 1
    return out / count
