
"""
tiff_processor.py

Functions to load and prepare TIFF terrain data.
"""

import tifffile
import numpy as np
import os

def load_tiff(tiff_path):
    """
    Loads TIFF terrain data and returns it as a NumPy array (float64).

    Args:
        tiff_path (str): Full path to TIFF file

    Returns:
        np.ndarray: Terrain data
    """
    terrain = tifffile.imread(tiff_path).astype(np.float64)
    return terrain
