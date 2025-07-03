
"""
hecras_loader.py

Utilities for loading HECRAS simulation data from HDF5 files.
"""

import h5py
import numpy as np
import os

def get_project_paths(prj_num, prj_name, plan_num):
    """
    Constructs necessary file paths for a given HEC-RAS simulation.

    Args:
        prj_num (str): Project ID
        prj_name (str): Project name
        plan_num (str): Plan number

    Returns:
        dict: Dictionary with keys: prj_path, terrain_path, plan_file_path, tiff_file_path
    """
    base_path = f"/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/HECRAS_Simulations_Results/prj_{prj_num}"
    plan_file = os.path.join(base_path, f"{prj_name}.p{plan_num}.hdf")
    tiff_file = os.path.join(base_path, "Terrains", f"terrain_{plan_num}.tif")

    return {
        "prj_path": base_path,
        "terrain_path": os.path.join(base_path, "Terrains"),
        "plan_file_path": plan_file,
        "tiff_file_path": tiff_file,
    }

def load_depth_and_coords(hdf_path):
    """
    Loads depth and cell center coordinates from HDF5 file.

    Args:
        hdf_path (str): Path to the HDF5 plan file

    Returns:
        dict: Dictionary with 'depth_vectors' and 'cells_center_coords'
    """
    with h5py.File(hdf_path, 'r') as f:
        results = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['Perimeter 1']
        geometry = f['Geometry']['2D Flow Areas']['Perimeter 1']

        depth_vectors = np.array(results['Cell Invert Depth'], dtype=np.float64)
        cell_coords = np.array(geometry['Cells Center Coordinate'], dtype=np.float64)

    return {
        "depth_vectors": depth_vectors,
        "cells_center_coords": cell_coords,
    }
