"""
Deep_Closure.py

This script implements the Deep Closure Model to predict future water depths over a large domain
using a trained deep learning model that operates on patches. The script handles boundary conditions,
data reshaping, patch processing, and iterative model inference.

Author: [Your Name]
Created: [Date]
"""

#%% Imports
import h5py
import numpy as np
import os
import tifffile
import copy
import torch
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from sklearn.metrics import mean_absolute_error

# Configure default font for plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman']

# Set global torch settings
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add function directory to path
# Append the relative path to the Functions folder (assumed to be in the same repo)
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'Functions'))
from TerrainDownsample_k11s1p0 import TerrainDownsample_k11s1p0  # the chosen terrain downsampler used
from arch_05 import arch_05  # the chosen deep learning architecture


#%% Deep_Closure Class
class Deep_Closure:
    """
    Deep Closure Model

    This class applies a trained patch-based deep learning model to predict future water
    depths over a large domain, accounting for unknown internal boundary conditions (BCs)
    using an iterative closure strategy.
    """

    def __init__(self,
                 prj_num='03',
                 prj_name='hecras_on_03',
                 plan_num='14',
                 cells_in_patch=32,
                 t=1002,
                 tolerance=1e-4,
                 delta_t=60):
        """
        Initialize the Deep_Closure class.

        Args:
            prj_num (str): Project number
            prj_name (str): HECRAS project name
            plan_num (str): Plan number for the simulation
            cells_in_patch (int): Number of cells in each patch
            t (int): Time index at which closure begins
            tolerance (float): Convergence tolerance for iterations
            delta_t (int): Time step in minutes
        """
        self.prj_num = prj_num
        self.prj_name = prj_name
        self.plan_num = plan_num

        # Define project paths
        self.prj_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/HECRAS_Simulations_Results/prj_{self.prj_num}'
        self.terrain_path = os.path.join(self.prj_path, 'Terrains')

        self.plan_file_name = f'{prj_name}.p{plan_num}.hdf'
        self.tiff_name = f'terrain_{plan_num}.tif'

        # Grid and patch settings
        self.meters_in_cell = 10  # Each simulation cell represents a 10x10 meter square
        self.cells_in_patch = cells_in_patch
        self.t = t
        self.delta_t = delta_t
        self.tolerance = tolerance

        # Flags and containers
        self.trimmed_1km_inwards = False
        self.num_patches_row_dict = {}
        self.num_patches_col_dict = {}
        self.num_patches_dict = {}

        # Load model and terrain downsampler
        self.model = arch_05().to(device)
        self.downsampler = TerrainDownsample_k11s1p0().to(device)

        # Patch sweeping order
        self.patch_orderings = ['A', 'B', 'C', 'D']

    def from_HDF_file(self):
        """
        Loads HECRAS output data from the specified HDF5 file.

        Returns:
            dict: A dictionary containing water depth and cell center coordinates.
        """
        from_HDF = {}
        os.chdir(self.prj_path)

        with h5py.File(self.plan_file_name, 'r') as f:
            Results = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['Perimeter 1']
            from_HDF['Invert_Depth'] = np.array(Results['Cell Invert Depth'], dtype=np.float64)
            from_HDF['Cells Center Coordinate'] = np.array(f['Geometry']['2D Flow Areas']['Perimeter 1']['Cells Center Coordinate'], dtype=np.float64)

        return from_HDF

    def populate_from_HDF(self):
        """
        Populates depth and coordinate matrices from the HDF5 simulation output.
        """
        os.chdir(self.prj_path)
        from_HDF_dict = self.from_HDF_file()

        self.depth_vectors = from_HDF_dict['Invert_Depth']  # shape: (timesteps, num_cells)
        self.cells_center_coords = from_HDF_dict['Cells Center Coordinate']  # shape: (num_cells, 2)

        # Placeholder for valid time indices (could be used for filtering smooth regions)
        valid_time_indices = []
        for t in range(len(self.depth_vectors) - self.delta_t):
            diff = self.depth_vectors[t] - self.depth_vectors[t + self.delta_t]
            if np.max(np.abs(diff)) < 1:
                valid_time_indices.append(t)

    def find_num_rows_cols_in_HECRAS(self):
        """
        Infers the number of rows and columns in the simulation grid based on cell coordinates.
        HECRAS includes an outer boundary of cells which are excluded here.
        """
        threshold = 1
        count_cols = sum(1 for row in self.cells_center_coords if abs(row[1] - self.cells_center_coords[0][1]) < threshold)
        self.num_cols = count_cols - 2  # exclude boundary

        count_rows = sum(1 for row in self.cells_center_coords if abs(row[0] - self.cells_center_coords[0][0]) < threshold)
        self.num_rows = count_rows - 2  # exclude boundary

        self.num_cells = self.num_rows * self.num_cols

    def one_matrix_depth(self):
        """
        Reshapes 1D depth vectors into 2D matrices for current and next time step.
        Optionally trims outer 1km if domain is large.
        """
        depth_vector = self.depth_vectors[self.t][:self.num_cells]
        depth_vector_next = self.depth_vectors[self.t + self.delta_t][:self.num_cells]

        new_shape = (self.num_rows, self.num_cols)
        self.depth_matrix = np.reshape(depth_vector, new_shape)
        self.depth_matrix_next = np.reshape(depth_vector_next, new_shape)

        # If domain is large, trim 100 cells (~1 km) from all sides
        if self.num_rows > 240 and self.num_cols > 240:
            self.trimmed_1km_inwards = True
            print('Trimming 1 km borders from depth matrices')
            self.depth_matrix = self.depth_matrix[100:-100, 100:-100]
            self.depth_matrix_next = self.depth_matrix_next[100:-100, 100:-100]
            self.num_rows -= 200
            self.num_cols -= 200
            self.num_cells = self.num_rows * self.num_cols

    def load_tiff_data(self):
        """
        Loads and optionally trims the terrain TIFF file.
        """
        os.chdir(self.terrain_path)
        self.tiff_data = tifffile.imread(self.tiff_name).astype(np.float64)

        if self.trimmed_1km_inwards:
            # Trim terrain to match trimmed depth domain
            self.tiff_data = self.tiff_data[1000:, 1000:]

    def calculate_num_patches_and_trim(self):
        """
        Calculates the number of patches in each direction based on the domain size.
        Also trims depth and terrain data to fit full patches only.
        """
        # Calculate size of each patch in meters and terrain points
        self.meters_in_patch = self.meters_in_cell * self.cells_in_patch
        self.tiff_points_in_patch = self.meters_in_patch + 1  # +1 for edges

        # Compute how many full patches fit in each direction
        self.num_patches_row_dict['A'] = (self.num_rows - 1) // self.cells_in_patch
        self.num_patches_col_dict['A'] = (self.num_cols - 1) // self.cells_in_patch
        self.num_patches_dict['A'] = self.num_patches_row_dict['A'] * self.num_patches_col_dict['A']

        # Trim domain to ensure full patches fit
        self.num_rows = self.num_patches_row_dict['A'] * self.cells_in_patch
        self.num_cols = self.num_patches_col_dict['A'] * self.cells_in_patch

        # Trim terrain TIFF to match trimmed depth matrices
        total_tiff_rows = self.num_rows * self.meters_in_cell + 1
        total_tiff_cols = self.num_cols * self.meters_in_cell + 1
        self.tiff_data = self.tiff_data[:total_tiff_rows, :total_tiff_cols]

        # Trim depth matrices accordingly
        self.depth_matrix = self.depth_matrix[:self.num_rows, :self.num_cols]
        self.depth_matrix_next = self.depth_matrix_next[:self.num_rows, :self.num_cols]

        # Save ground truth for comparison after closure
        self.saved_true_matrix_next = copy.deepcopy(self.depth_matrix_next)

        # Replace unknown internal BCs with initial guess (ICs)
        self.depth_matrix_next = self.previous_depth_internal(self.depth_matrix_next)
        self.depth_matrix_next_dummy = copy.deepcopy(self.depth_matrix_next)

    def previous_depth_internal(self, matrix):
        """
        Sets internal boundary conditions for a depth matrix.

        Args:
            matrix (np.ndarray): The matrix to modify

        Returns:
            np.ndarray: Matrix with internal BCs initialized to current depths (ICs)
        """
        result = matrix.copy()
        result[1:-1, 1:-1] = self.depth_matrix[1:-1, 1:-1]
        return result

    def iterate_closure(self):
        """
        Performs the iterative forward passes over the entire domain using different patch orderings.
        Updates internal boundary conditions until predictions converge within a set tolerance.
        """
        print("Starting closure iterations...")
        num_iters = 0
        max_diff = float('inf')

        while max_diff > self.tolerance and num_iters < 50:
            previous_prediction = self.depth_matrix_next.copy()

            for ordering in self.patch_orderings:
                # Placeholder: here you would extract patches, run inference, and update matrix
                # In your implementation, patch ordering A/B/C/D is handled by patch extractors.
                pass

            max_diff = np.max(np.abs(previous_prediction - self.depth_matrix_next))
            print(f"Iteration {num_iters}, Max Δ = {max_diff:.6f}")
            num_iters += 1

        print("Closure iterations complete.")

    def evaluate_RAE(self):
        """
        Evaluates the prediction accuracy using Relative Absolute Error (RAE).

        Returns:
            float: RAE score
        """
        abs_error = np.abs(self.saved_true_matrix_next - self.depth_matrix_next)
        abs_reference = np.abs(self.saved_true_matrix_next)
        return np.sum(abs_error) / (np.sum(abs_reference) + 1e-8)

    def run(self):
        """
        Executes the full Deep Closure process.
        """
        self.populate_from_HDF()
        self.find_num_rows_cols_in_HECRAS()
        self.one_matrix_depth()
        self.load_tiff_data()
        self.calculate_num_patches_and_trim()
        self.iterate_closure()
        rae = self.evaluate_RAE()
        print(f"Final RAE: {rae:.6f}")


#%% Example Usage
if __name__ == "__main__":
    closure = Deep_Closure(
        prj_num='03',
        prj_name='hecras_on_03',
        plan_num='14',
        cells_in_patch=32,
        t=1002,
        tolerance=1e-4,
        delta_t=60
    )
    closure.run()
