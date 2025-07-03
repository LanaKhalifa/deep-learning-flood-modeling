"""
hecras_loader.py

This module contains functions to load terrain data (from TIFF files) and
simulation results (from HDF5 files) used in the Deep Closure Model.

Author: [Your Name]
Created: [Date]
"""

import tifffile
import h5py
import numpy as np
import os

def load_terrain(tiff_path):
    """
    Load the terrain elevation data from a TIFF file.

    Parameters:
    - tiff_path (str): Full path to the .tif terrain file.

    Returns:
    - np.ndarray: 2D array of terrain elevations.
    """
    terrain = tifffile.imread(tiff_path).astype(np.float64)
    return terrain

def load_simulation_data(hdf5_path):
    """
    Load required simulation data from a HEC-RAS HDF5 file.

    Parameters:
    - hdf5_path (str): Full path to the .hdf file.

    Returns:
    - dict: A dictionary with required datasets extracted from the file.
    """
    data = {}
    with h5py.File(hdf5_path, 'r') as f:
        # Example: Load water depth and WSE datasets
        # NOTE: You can add more keys if needed
        data['Depth'] = f['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Depth'][:]
        data['WSE'] = f['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Water Surface'][:]
        # You can add more datasets as needed, for example velocities, etc.

    return data
