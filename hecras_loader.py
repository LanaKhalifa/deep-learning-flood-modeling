import os
import h5py
import tifffile
import numpy as np

def load_hdf_data(hdf_path: str, plan_file: str) -> dict:
    """
    Loads HECRAS output data from the specified HDF5 file.

    Args:
        hdf_path (str): Path to the directory containing the HDF5 file.
        plan_file (str): Filename of the HDF5 simulation results.

    Returns:
        dict: Dictionary with depth time series and cell center coordinates.
    """
    file_path = os.path.join(hdf_path, plan_file)
    with h5py.File(file_path, 'r') as f:
        results = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['Perimeter 1']
        invert_depth = np.array(results['Cell Invert Depth'], dtype=np.float64)

        coords = np.array(
            f['Geometry']['2D Flow Areas']['Perimeter 1']['Cells Center Coordinate'],
            dtype=np.float64
        )

    return {
        'Invert_Depth': invert_depth,
        'Cells Center Coordinate': coords
    }

def load_tiff_data(tiff_path: str, tiff_file: str) -> np.ndarray:
    """
    Loads the terrain TIFF file.

    Args:
        tiff_path (str): Path to the directory containing the TIFF file.
        tiff_file (str): Filename of the terrain raster.

    Returns:
        np.ndarray: 2D terrain elevation array.
    """
    file_path = os.path.join(tiff_path, tiff_file)
    terrain = tifffile.imread(file_path).astype(np.float64)
    return terrain
