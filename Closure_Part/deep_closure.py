
"""
deep_closure.py

Implements the DeepClosure class to apply trained DL model on large HECRAS domains
using patch-based inference and iterative closure on internal boundary conditions.
"""

import os
import numpy as np
import torch
import copy
from sklearn.metrics import mean_absolute_error

from config import *
from hecras_loader import get_project_paths, load_depth_and_coords
from tiff_processor import load_tiff
from patch_utils import make_depth_patches, make_tiff_patches
from model_utils import load_trained_model, prepare_input_tensor, prepare_tiff_tensor, run_model, fill_internal_with
from plot_utils import plot_convergence, plot_comparison_matrices


class DeepClosure:
    def __init__(self, prj_num, prj_name, plan_num, t, tolerance=1e-3):
        self.prj_num = prj_num
        self.prj_name = prj_name
        self.plan_num = plan_num
        self.t = t
        self.tolerance = tolerance

        # File paths
        self.paths = get_project_paths(prj_num, prj_name, plan_num)

        # Load model
        self.model = load_trained_model()

        # Output storage
        self.L1_val = None
        self.RAE_val = None
        self.depth_matrix_next = None
        self.depth_matrix_next_dummy = None
        self.saved_true_matrix_next = None

    def load_data(self):
        # Load depth and coordinate data
        data = load_depth_and_coords(self.paths["plan_file_path"])
        self.depth_vectors = data["depth_vectors"]
        self.cell_coords = data["cells_center_coords"]

        # Infer domain shape
        self.num_rows, self.num_cols = self._infer_matrix_shape()
        self.num_cells = self.num_rows * self.num_cols

        # Load depth matrices
        d_curr = self.depth_vectors[self.t][:self.num_cells]
        d_next = self.depth_vectors[self.t + DELTA_T][:self.num_cells]
        self.depth_matrix = np.reshape(d_curr, (self.num_rows, self.num_cols))
        self.depth_matrix_next = np.reshape(d_next, (self.num_rows, self.num_cols))
        self.saved_true_matrix_next = copy.deepcopy(self.depth_matrix_next)
        self.depth_matrix_next_dummy = copy.deepcopy(self.depth_matrix_next)
        self.depth_matrix_next = fill_internal_with(self.depth_matrix_next, self.depth_matrix)

        # Load terrain
        self.tiff_data = load_tiff(self.paths["tiff_file_path"])

        # Compute patch counts and crop matrices
        self._calculate_patches_and_trim()

    def _infer_matrix_shape(self):
        threshold = 1
        count_cols = sum(1 for row in self.cell_coords if abs(row[1] - self.cell_coords[0][1]) < threshold) - 2
        count_rows = sum(1 for row in self.cell_coords if abs(row[0] - self.cell_coords[0][0]) < threshold) - 2
        return count_rows, count_cols

    def _calculate_patches_and_trim(self):
        self.num_patches_row = (self.num_rows - 1) // CELLS_IN_PATCH
        self.num_patches_col = (self.num_cols - 1) // CELLS_IN_PATCH
        self.num_patches = self.num_patches_row * self.num_patches_col

        # Update shape and crop
        self.num_rows = self.num_patches_row * CELLS_IN_PATCH
        self.num_cols = self.num_patches_col * CELLS_IN_PATCH

        self.depth_matrix = self.depth_matrix[:self.num_rows, :self.num_cols]
        self.depth_matrix_next = self.depth_matrix_next[:self.num_rows, :self.num_cols]
        self.depth_matrix_next_dummy = self.depth_matrix_next_dummy[:self.num_rows, :self.num_cols]
        self.saved_true_matrix_next = self.saved_true_matrix_next[:self.num_rows, :self.num_cols]

        # Cut terrain
        patch_width_m = METERS_IN_CELL * CELLS_IN_PATCH
        points_in_patch = patch_width_m + 1
        self.tiff_data = self.tiff_data[
            :self.num_patches_row * METERS_IN_CELL * CELLS_IN_PATCH + 1,
            :self.num_patches_col * METERS_IN_CELL * CELLS_IN_PATCH + 1,
        ]

    def run_closure_loop(self):
        self.load_data()

        next_old_diffs = []
        true_pred_diffs = []

        dummy_mae = mean_absolute_error(self.saved_true_matrix_next.flatten(), self.depth_matrix_next_dummy.flatten())
        true_pred_diff = mean_absolute_error(self.saved_true_matrix_next.flatten(), self.depth_matrix_next.flatten())
        true_pred_diffs.append(true_pred_diff)

        # First pass
        old_matrix = copy.deepcopy(self.depth_matrix_next)
        self._predict_and_update()
        next_old_diff = mean_absolute_error(old_matrix.flatten(), self.depth_matrix_next.flatten())
        next_old_diffs.append(next_old_diff)
        true_pred_diff = mean_absolute_error(self.saved_true_matrix_next.flatten(), self.depth_matrix_next.flatten())
        true_pred_diffs.append(true_pred_diff)

        # Iterative closure
        while next_old_diff > self.tolerance and len(next_old_diffs) < 30:
            old_matrix = copy.deepcopy(self.depth_matrix_next)
            self._predict_and_update()
            next_old_diff = mean_absolute_error(old_matrix.flatten(), self.depth_matrix_next.flatten())
            next_old_diffs.append(next_old_diff)
            true_pred_diff = mean_absolute_error(self.saved_true_matrix_next.flatten(), self.depth_matrix_next.flatten())
            true_pred_diffs.append(true_pred_diff)

        # Final metrics
        self.L1_val = true_pred_diffs[-1]
        num = np.linalg.norm(self.saved_true_matrix_next - self.depth_matrix_next, ord=1)
        denom = np.linalg.norm(self.depth_matrix_next, ord=1)
        self.RAE_val = num / denom

        # Plot convergence
        plot_convergence(next_old_diffs, true_pred_diffs, dummy_mae,
            f"{BASE_PROJECT_PATH}/Closure/Closure_Loop/{self.prj_num}_{self.plan_num}_{self.t}.png")

    def _predict_and_update(self):
        # Create input patches
        depth_patches = make_depth_patches(self.depth_matrix, CELLS_IN_PATCH)
        terrain_patches = make_tiff_patches(self.tiff_data, METERS_IN_CELL * CELLS_IN_PATCH + 1)
        target_patches = make_depth_patches(self.depth_matrix_next, CELLS_IN_PATCH)

        # Create tensors
        x_depth = prepare_input_tensor(depth_patches, target_patches)
        x_terr = prepare_tiff_tensor(terrain_patches)

        # Predict delta
        predicted_delta = run_model(self.model, x_terr, x_depth)
        predicted_depth = predicted_delta + depth_patches

        # Reconstruct
        self._reconstruct_from_patches(predicted_depth)

    def _reconstruct_from_patches(self, patches):
        out = np.zeros((self.num_rows, self.num_cols), dtype=np.float64)
        k = 0
        for i in range(self.num_patches_row):
            for j in range(self.num_patches_col):
                r0 = i * CELLS_IN_PATCH
                c0 = j * CELLS_IN_PATCH
                out[r0:r0+CELLS_IN_PATCH, c0:c0+CELLS_IN_PATCH] = patches[k]
                k += 1
        self.depth_matrix_next[2:-2, 2:-2] = out[2:-2, 2:-2]

    def plot_all_matrices(self):
        path = f"{BASE_PROJECT_PATH}/Closure/9_Maps/{self.prj_num}_{self.plan_num}_{self.t}.png"
        plot_comparison_matrices(self.depth_matrix_next, self.depth_matrix_next_dummy, self.saved_true_matrix_next,
                                 self.L1_val, self.RAE_val, path)
