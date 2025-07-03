"""
closure_model.py

This script defines the Deep_Closure class which applies a trained deep learning
model to iteratively predict future water depths across a large domain.

Author: [Your Name]
Created: [Date]
"""

import os
import h5py
import tifffile
import numpy as np
import copy
import torch
from sklearn.metrics import mean_absolute_error

# Local modules (assumes they are in the same directory or in PYTHONPATH)
from hdf_tiff_loader import load_terrain, load_simulation_data
from matrix_patcher import extract_patches, rebuild_from_patches

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Deep_Closure:
    def __init__(self,
                 prj_num='03',
                 prj_name='hecras_on_03',
                 plan_num='14',
                 cells_in_patch=32,
                 t=1002,
                 tolerance=1e-4,
                 delta_t=60):
        """
        Initializes the Deep Closure model.

        Parameters:
        - prj_num, prj_name, plan_num: Identifiers for HEC-RAS project and simulation plan
        - cells_in_patch: Patch size for DL model input
        - t: Current time index for simulation
        - tolerance: Convergence threshold
        - delta_t: Prediction interval (in minutes)
        """

        self.prj_num = prj_num
        self.prj_name = prj_name
        self.plan_num = plan_num

        self.prj_path = f'./HECRAS_Simulations_Results/prj_{self.prj_num}'
        self.terrain_path = os.path.join(self.prj_path, 'Terrains')

        self.plan_file_name = os.path.join(self.prj_path, f'{prj_name}.p{plan_num}.hdf')
        self.tiff_name = os.path.join(self.terrain_path, f'terrain_{plan_num}.tif')

        self.meters_in_cell = 10
        self.cells_in_patch = cells_in_patch
        self.meters_in_patch = self.meters_in_cell * self.cells_in_patch
        self.t = t
        self.delta_t = delta_t
        self.tolerance = tolerance

        # Patch tracking for 4 sweep directions
        self.patches_depth_next_dict = {k: None for k in 'ABCD'}
        self.patches_depth_dict = {k: None for k in 'ABCD'}
        self.patches_true_depth_next_dict = {k: None for k in 'ABCD'}
        self.patches_tiff_dict = {k: None for k in 'ABCD'}
        self.num_patches_dict = {k: None for k in 'ABCD'}
        self.num_patches_row_dict = {k: None for k in 'ABCD'}
        self.num_patches_col_dict = {k: None for k in 'ABCD'}

        self.trimmed_1km_inwards = False
        self.saved_initial_BD = None

    def from_HDF_file(self):
        with h5py.File(self.plan_file_name, 'r') as f:
            results = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['Perimeter 1']
            invert_depth = np.array(results['Cell Invert Depth'], dtype=np.float64)
            coords = np.array(f['Geometry']['2D Flow Areas']['Perimeter 1']['Cells Center Coordinate'], dtype=np.float64)
            return {'Invert_Depth': invert_depth, 'Cells Center Coordinate': coords}

    def populate_from_HDF(self):
        os.chdir(self.prj_path)
        from_HDF_dict = self.from_HDF_file()
        self.depth_vectors = from_HDF_dict['Invert_Depth']
        self.cells_center_coords = from_HDF_dict['Cells Center Coordinate']

        # Logic for valid time indices (filtered by small change)
        self.valid_time_indices = []
        for t in range(len(self.depth_vectors) - self.delta_t):
            diff = self.depth_vectors[t] - self.depth_vectors[t + self.delta_t]
            if np.max(np.abs(diff)) < 1:
                self.valid_time_indices.append(t)

    def find_num_rows_cols_in_HECRAS(self):
        threshold = 1
        count = sum(1 for row in self.cells_center_coords if abs(row[1] - self.cells_center_coords[0][1]) < threshold)
        self.num_cols = count - 2
        count = sum(1 for row in self.cells_center_coords if abs(row[0] - self.cells_center_coords[0][0]) < threshold)
        self.num_rows = count - 2
        self.num_cells = self.num_rows * self.num_cols

    def one_matrix_depth(self):
        # Take the t and t+delta_t slices and reshape into matrix
        vec_now = self.depth_vectors[self.t][:self.num_cells]
        vec_next = self.depth_vectors[self.t + self.delta_t][:self.num_cells]
        shape = (self.num_rows, self.num_cols)
        self.depth_matrix = np.reshape(vec_now, shape)
        self.depth_matrix_next = np.reshape(vec_next, shape)

        # Optional trimming
        if self.num_rows > 240 and self.num_cols > 240:
            self.trimmed_1km_inwards = True
            self.depth_matrix = self.depth_matrix[100:-100, 100:-100]
            self.depth_matrix_next = self.depth_matrix_next[100:-100, 100:-100]
            self.num_rows -= 200
            self.num_cols -= 200
            self.num_cells = self.num_rows * self.num_cols
