
"""
patch_utils.py

Utility functions for slicing depth and terrain matrices into patches.
"""

import numpy as np
import copy

def generate_patch_indices(num_rows, num_cols, patch_size):
    """
    Generate top-left corner indices for patch slicing.

    Args:
        num_rows (int): Total rows in matrix
        num_cols (int): Total columns in matrix
        patch_size (int): Size of square patch

    Returns:
        list of tuples: (row, col) indices
    """
    row_indices = np.arange(0, num_rows, patch_size)
    col_indices = np.arange(0, num_cols, patch_size)
    rows, cols = np.meshgrid(row_indices, col_indices)
    return np.column_stack((rows.ravel(), cols.ravel()))

def make_depth_patches(matrix, patch_size):
    """
    Slice a 2D depth matrix into square patches.

    Args:
        matrix (np.ndarray): 2D array (depth)
        patch_size (int): Size of each patch

    Returns:
        np.ndarray: 3D array of patches (N, patch_size, patch_size)
    """
    num_rows = matrix.shape[0] // patch_size
    num_cols = matrix.shape[1] // patch_size
    indices = generate_patch_indices(num_rows * patch_size, num_cols * patch_size, patch_size)

    patches = np.zeros((len(indices), patch_size, patch_size), dtype=np.float64)
    for i, (r, c) in enumerate(indices):
        patches[i] = matrix[r:r+patch_size, c:c+patch_size]

    return patches

def make_tiff_patches(matrix, patch_size):
    """
    Slice a 2D terrain TIFF matrix into overlapping patches.

    Args:
        matrix (np.ndarray): 2D array (terrain)
        patch_size (int): Size of the TIFF patch in points

    Returns:
        np.ndarray: 3D array of normalized patches (N, patch_size, patch_size)
    """
    stride = patch_size - 1
    num_rows = (matrix.shape[0] - 1) // stride
    num_cols = (matrix.shape[1] - 1) // stride

    patches = np.zeros((num_rows * num_cols, patch_size, patch_size), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            row = i * stride
            col = j * stride
            patch = matrix[row:row+patch_size, col:col+patch_size]
            patches[i * num_cols + j] = patch - np.min(patch)
    return patches
