"""
This script extracts and processes flood simulation data from HDF files and TIFF terrain files
for further use in deep learning models.
"""

# Standard library
import os
import copy
import pickle
import random

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tifffile
from scipy import ndimage
from scipy.spatial.distance import cdist


class PatchExtractorProcessor:
    """
    PatchExtractorProcessor

    Processes raw flood simulation outputs (HDF and TIFF files) into structured
    training data for deep learning.

    Responsibilities:
    - Load water depth and terrain data from HDF and TIFF
    - Trim and align depth and terrain grids
    - Extract patches (standard and dual)
    - Apply data augmentations (flip and rotate)
    - Remove dry/irrelevant patches
    - Store output in a sample-ready format: {'terrain', 'depth', 'depth_next'}
    """
    def __init__(self,
                 prj_num: str,
                 prj_name: str, 
                 plan_num: str,
                 plot: bool = False):
        """
        Initializes a PatchExtractorProcessor object to extract and process simulation data.

        Args:
            prj_num (str): Custom project number (e.g., "03", "04", etc.)
            prj_name (str): Project name as used in HECRAS.
            plan_num (str): Plan number for the simulation.
        """
        # Project identifiers
        self.prj_num = prj_num
        self.prj_name = prj_name
        self.plan_num = plan_num

        # File paths (relative to GitHub root)
        self.prj_path = os.path.join('HECRAS_Simulations_Results', f'prj_{prj_num}')
        self.terrain_path = os.path.join(self.prj_path, 'Terrains')
        self.plan_file_name = f'{prj_name}.p{plan_num}.hdf'
        self.tiff_name = f'terrain_{plan_num}.tif'

        # Simulation settings
        self.k = 4  # Number of snapshots taken from each simulation
        self.delta_t = 60  # Time step between input and output in minutes
        self.closest_indices = [0, 70, 140, 210]  # Time points used
        self.meters_in_cell = 10  # Cell resolution set in HECRAS
        self.cells_in_patch = 32  # Each sample is a 32x32 patch

        # Placeholders for internal data
        self.depth_vectors = None
        self.cells_center_coords = None
        self.depth_matrices = None
        self.num_cols = None
        self.num_rows = None
        self.num_cells = None
        self.k_depth_matrices = None
        self.k_depth_matrices_next = None
        self.tiff_data = None
        self.tiff_data_dual = None
        self.meters_in_patch = None
        self.tiff_points_in_patch = None
        self.num_patches_row = None
        self.num_patches_col = None
        self.num_patches = None
        self.num_patches_row_dual = None
        self.num_patches_col_dual = None
        self.num_patches_dual = None
        self.patches_tiff = None
        self.patches_tiff_dual = None
        self.patches_depth = None
        self.patches_depth_next = None
        self.patches_depth_dual = None
        self.patches_depth_next_dual = None

        # Miscellaneous
        self.cluster_counter = 0
        self.trimmed_1km_inwards = False
        self.database = {'terrain': [], 'depth': [], 'depth_next': []}

        #plotting
        self.plot = plot  # Enable plotting manually if needed

    def from_HDF_file(self):
        """
        Extracts cell center coordinates and invert depth values from the HDF5 plan file.
        Returns:
            dict: A dictionary with keys:
                  - 'Cells Center Coordinate': ndarray of cell centers
                  - 'Invert_Depth': ndarray of invert depth values
        """
        hdf_path = os.path.join(self.prj_path, self.plan_file_name)

        with h5py.File(hdf_path, 'r') as f:
            data = {}

            # Geometry: Cell center coordinates
            data['Cells Center Coordinate'] = np.array(
                f['Geometry']['2D Flow Areas']['Perimeter 1']['Cells Center Coordinate']
            )

            # Results: Invert depth values over time
            results_group = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']\
                              ['Unsteady Time Series']['2D Flow Areas']['Perimeter 1']
            data['Invert_Depth'] = np.array(results_group['Cell Invert Depth'])

        return data

    def populate_from_HDF(self):
        """
        Populates depth vectors and cell center coordinates from the HDF file.
        """
        from_HDF_dict = self.from_HDF_file()

        # Depth vectors: 2D array [time, cell]
        self.depth_vectors = from_HDF_dict['Invert_Depth']

        # Cell center coordinates: 2D array [cell, (x, y)]
        self.cells_center_coords = from_HDF_dict['Cells Center Coordinate']


    def find_num_rows_cols_in_HECRAS(self):
        """
        HECRAS does not provide the number of rows and columns of the 2D grid.
        This method calculates it based on spatial patterns in cell center coordinates,
        then reshapes the depth vectors accordingly, and optionally trims 1 km from edges.
        """

        threshold = 1  # Tolerance for coordinate similarity (in meters)

        # Estimate number of columns: how many cells share the same y-coordinate as the first cell
        self.num_cols = sum(1 for row in self.cells_center_coords if abs(row[1] - self.cells_center_coords[0][1]) < threshold) - 2
        # Estimate number of rows: how many cells share the same x-coordinate as the first cell
        self.num_rows = sum(1 for row in self.cells_center_coords if abs(row[0] - self.cells_center_coords[0][0]) < threshold) - 2
        self.num_cells = self.num_rows * self.num_cols

        # Extract k depth vectors at selected time steps and their corresponding future states
        k_depth_vectors = np.zeros((self.k, self.depth_vectors.shape[1]))
        k_depth_vectors_next = np.zeros((self.k, self.depth_vectors.shape[1])) # next is 60 mins ahead

        for i, idx in enumerate(self.closest_indices):
            k_depth_vectors[i] = self.depth_vectors[idx]
            k_depth_vectors_next[i] = self.depth_vectors[idx + self.delta_t]

        # Keep only valid cells (remove padding that HECRAS uses, called fictitious cells)
        k_depth_vectors = k_depth_vectors[:, :self.num_cells]
        k_depth_vectors_next = k_depth_vectors_next[:, :self.num_cells]

        new_shape = (self.k, self.num_rows, self.num_cols)

        # Reshape vectors into matrices (row-major order, same as HECRAS)
        self.k_depth_matrices = np.reshape(k_depth_vectors, new_shape)
        self.k_depth_matrices_next = np.reshape(k_depth_vectors_next, new_shape)

        # Optional trimming: Remove 1 km (100 cells) from each edge if domain is large to recude artificats found near boundaries 
        if self.num_rows > 240 and self.num_cols > 240: 
            self.trimmed_1km_inwards = True
            print('Trimming water depths...')
            self.k_depth_matrices = self.k_depth_matrices[:, 100:-100, 100:-100]
            self.k_depth_matrices_next = self.k_depth_matrices_next[:, 100:-100, 100:-100]
            self.num_rows -= 200
            self.num_cols -= 200
            self.num_cells = self.num_rows * self.num_cols

    def plot_depth_maps(self):
        """
        Plots the water depth maps for each time snapshot in self.k_depth_matrices.
        Only runs if self.plot is True.
        """
        if not self.plot:
            return  # Skip plotting unless explicitly enabled

        plt.figure(figsize=(6, 6))

        for i in range(self.k):
            plt.clf()
            plt.imshow(self.k_depth_matrices[i], cmap='Blues', vmin=0, vmax=3)
            plt.colorbar(label='Water Depth [m]')
            plt.title(f'Project {self.prj_num} | Plan {self.plan_num} | Snapshot {i}')
            plt.pause(1)

        plt.show()

    def load_tiff_data(self):
        """
        Loads the terrain TIFF file into self.tiff_data.
        If trimming is enabled, trims 1 km (1000 TIFF points) from each side.
        Optionally plots the terrain if self.plot is True.

        """
        tiff_path = os.path.join(self.terrain_path, self.tiff_name)
        self.tiff_data = tifffile.imread(tiff_path)

        # Trim 1 km (1000 TIFF pixels) from edges if required
        if self.trimmed_1km_inwards:
            # Note: We trim only from the top and left to shift the origin of the TIFF.
            # This ensures alignment with the trimmed water depth grid, as terrain patches
            # will later be extracted according to the water depth cells.
            self.tiff_data = self.tiff_data[1000:, 1000:]

        # Optional plot
        if self.plot:
            plt.imshow(self.tiff_data, cmap='terrain')
            plt.colorbar()
            plt.title(f'Terrain of prj_{self.prj_num} - plan_{self.plan_num}')
            plt.show()
            
    def calculate_num_patches(self):
        """
        Calculates the number of patches (standard and dual) for depth and terrain data,
        based on the number of cells and the patch size.
        """
        # Calculate patch size in meters and corresponding TIFF resolution
        self.meters_in_patch = self.meters_in_cell * self.cells_in_patch
        self.tiff_points_in_patch = self.meters_in_patch + 1  # +1 accounts for edges

        # Standard patches: exclude last row/column to avoid non-square patches
        self.num_patches_row = (self.num_rows - 1) // self.cells_in_patch
        self.num_patches_col = (self.num_cols - 1) // self.cells_in_patch
        self.num_patches = self.num_patches_row * self.num_patches_col

        # Dual patches: start halfway into the grid (in both dimensions)
        offset = self.cells_in_patch // 2
        self.num_patches_row_dual = (self.num_rows - offset - 1) // self.cells_in_patch
        self.num_patches_col_dual = (self.num_cols - offset - 1) // self.cells_in_patch
        self.num_patches_dual = self.num_patches_row_dual * self.num_patches_col_dual

    def tiff_patches(self):
        """
        Extracts terrain patches (standard and dual) from the TIFF terrain data.
        Each patch covers tiff_points_in_patch × tiff_points_in_patch points.
        """
        # --- Standard patches ---
        self.patches_tiff = np.zeros((self.num_patches, self.tiff_points_in_patch, self.tiff_points_in_patch))
        row_indices = np.arange(self.num_patches_row) * (self.tiff_points_in_patch - 1)
        col_indices = np.arange(self.num_patches_col) * (self.tiff_points_in_patch - 1)
        rows, cols = np.meshgrid(row_indices, col_indices)
        indices = np.column_stack((rows.ravel(), cols.ravel()))

        for i, (row, col) in enumerate(indices):
            patch = self.tiff_data[row:row + self.tiff_points_in_patch, col:col + self.tiff_points_in_patch]
            self.patches_tiff[i] = copy.deepcopy(patch)

        # --- Dual patches ---
        self.patches_tiff_dual = np.zeros((self.num_patches_dual, self.tiff_points_in_patch, self.tiff_points_in_patch))

        # Offset the origin by half a patch (in meters → pixels)
        tiff_start_offset = self.meters_in_patch // 2
        self.tiff_data_dual = self.tiff_data[tiff_start_offset:, tiff_start_offset:]

        row_indices = np.arange(self.num_patches_row_dual) * (self.tiff_points_in_patch - 1)
        col_indices = np.arange(self.num_patches_col_dual) * (self.tiff_points_in_patch - 1)
        rows, cols = np.meshgrid(row_indices, col_indices)
        indices = np.column_stack((rows.ravel(), cols.ravel()))

        for i, (row, col) in enumerate(indices):
            patch = self.tiff_data_dual[row:row + self.tiff_points_in_patch, col:col + self.tiff_points_in_patch]
            self.patches_tiff_dual[i] = copy.deepcopy(patch)

    def depth_patches(self, k_index):
        """
        Extracts depth and depth_next patches from the water depth matrices
        for the given time index (k_index).
        Uses standard patches for even indices and dual patches for odd indices.
        
        Args:
            k_index (int): Time index (0, 1, 2, or 3)
        """
        if k_index % 2 == 0:
            # --- Standard patches ---
            self.patches_depth = np.zeros((self.num_patches, self.cells_in_patch, self.cells_in_patch))
            self.patches_depth_next = np.zeros((self.num_patches, self.cells_in_patch, self.cells_in_patch))

            row_indices = np.arange(self.num_patches_row) * self.cells_in_patch
            col_indices = np.arange(self.num_patches_col) * self.cells_in_patch
            rows, cols = np.meshgrid(row_indices, col_indices)
            indices = np.column_stack((rows.ravel(), cols.ravel()))

            for i, (row, col) in enumerate(indices):
                patch = self.k_depth_matrices[k_index, row:row + self.cells_in_patch, col:col + self.cells_in_patch]
                patch_next = self.k_depth_matrices_next[k_index, row:row + self.cells_in_patch, col:col + self.cells_in_patch]

                self.patches_depth[i] = copy.deepcopy(patch)
                self.patches_depth_next[i] = copy.deepcopy(patch_next)

        else:
            # --- Dual patches ---
            self.patches_depth_dual = np.zeros((self.num_patches_dual, self.cells_in_patch, self.cells_in_patch))
            self.patches_depth_next_dual = np.zeros((self.num_patches_dual, self.cells_in_patch, self.cells_in_patch))

            offset = self.cells_in_patch // 2
            k_depth_dual = self.k_depth_matrices[:, offset:, offset:]
            k_depth_next_dual = self.k_depth_matrices_next[:, offset:, offset:]

            row_indices = np.arange(self.num_patches_row_dual) * self.cells_in_patch
            col_indices = np.arange(self.num_patches_col_dual) * self.cells_in_patch
            rows, cols = np.meshgrid(row_indices, col_indices)
            indices = np.column_stack((rows.ravel(), cols.ravel()))

            for i, (row, col) in enumerate(indices):
                patch = k_depth_dual[k_index, row:row + self.cells_in_patch, col:col + self.cells_in_patch]
                patch_next = k_depth_next_dual[k_index, row:row + self.cells_in_patch, col:col + self.cells_in_patch]

                self.patches_depth_dual[i] = copy.deepcopy(patch)
                self.patches_depth_next_dual[i] = copy.deepcopy(patch_next)

    def add_samples(self, patches_depth, patches_depth_next, patches_tiff, dual):
        """
        Adds depth, depth_next, and terrain patches to the database.
        Also displays one random sample as a visual check.
        """
        patches_depth = copy.deepcopy(patches_depth)
        patches_depth_next = copy.deepcopy(patches_depth_next)
        patches_tiff = copy.deepcopy(patches_tiff)

        self.cluster_counter += 1  # Track how many augmentation sets have been added

        for patch_depth, patch_depth_next, patch_terrain in zip(patches_depth, patches_depth_next, patches_tiff):
            sample = {
                'terrain': patch_terrain,
                'depth': patch_depth,
                'depth_next': patch_depth_next
            }
            for key, value in sample.items():
                self.database[key].append(value)

        # Plot a random example (for visual inspection)
        if len(patches_depth) > 1:
            idx = random.randint(0, len(patches_depth) - 1)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(patches_depth[idx], cmap='viridis')
            axs[0].set_title('Depth (32x32)')
            axs[1].imshow(patches_depth_next[idx], cmap='viridis')
            axs[1].set_title('Depth Next (32x32)')
            axs[2].imshow(patches_tiff[idx], cmap='terrain')
            axs[2].set_title('Terrain (321x321)')
            plt.tight_layout()
            plt.show()

    def flip(self, dual, side):
        """
        Flips depth and terrain patches horizontally or vertically.
        Returns flipped TIFF patches only (for rotation).
        """
        if side == 'horizontal':
            if not dual:
                self.patches_depth = np.fliplr(self.patches_depth)
                self.patches_depth_next = np.fliplr(self.patches_depth_next)
                patches_tiff_flipped = np.fliplr(copy.deepcopy(self.patches_tiff))
            else:
                self.patches_depth_dual = np.fliplr(self.patches_depth_dual)
                self.patches_depth_next_dual = np.fliplr(self.patches_depth_next_dual)
                patches_tiff_flipped = np.fliplr(copy.deepcopy(self.patches_tiff_dual))

        elif side == 'vertical':
            if not dual:
                self.patches_depth = np.flip(self.patches_depth, axis=2)
                self.patches_depth_next = np.flip(self.patches_depth_next, axis=2)
                patches_tiff_flipped = np.flip(copy.deepcopy(self.patches_tiff), axis=2)
            else:
                self.patches_depth_dual = np.flip(self.patches_depth_dual, axis=2)
                self.patches_depth_next_dual = np.flip(self.patches_depth_next_dual, axis=2)
                patches_tiff_flipped = np.flip(copy.deepcopy(self.patches_tiff_dual), axis=2)

        return patches_tiff_flipped

    def rotate(self, patches_tiff_flipped, dual, angle=0):
        """
        Rotates the flipped depth and terrain patches by a given angle.
        Returns rotated TIFF patches only.
        """
        if not dual:
            self.patches_depth = ndimage.rotate(self.patches_depth, angle, reshape=False, axes=(1, 2))
            self.patches_depth_next = ndimage.rotate(self.patches_depth_next, angle, reshape=False, axes=(1, 2))
            patches_tiff_rotated = ndimage.rotate(patches_tiff_flipped, angle, reshape=False, axes=(1, 2))
        else:
            self.patches_depth_dual = ndimage.rotate(self.patches_depth_dual, angle, reshape=False, axes=(1, 2))
            self.patches_depth_next_dual = ndimage.rotate(self.patches_depth_next_dual, angle, reshape=False, axes=(1, 2))
            patches_tiff_rotated = ndimage.rotate(patches_tiff_flipped, angle, reshape=False, axes=(1, 2))

        return patches_tiff_rotated


    def populate_from_all_k_indices(self):
        """
        Applies flipping and rotation augmentations for all 4 k_indices (0 to 3),
        then adds corresponding depth and terrain patches to the database.
        """
        print(f'Now in prj_{self.prj_num} plan_{self.plan_num}')

        for k_index in [0, 1, 2, 3]:
            self.depth_patches(k_index=k_index)

            if k_index == 0:
                self.add_samples(self.patches_depth, self.patches_depth_next, self.patches_tiff, dual=False)

            elif k_index == 1:
                patches_flipped = self.flip(dual=True, side='horizontal')
                patches_rotated = self.rotate(patches_flipped, dual=True, angle=90)
                self.add_samples(self.patches_depth_dual, self.patches_depth_next_dual, patches_rotated, dual=True)

            elif k_index == 2:
                patches_flipped = self.flip(dual=False, side='vertical')
                patches_rotated = self.rotate(patches_flipped, dual=False, angle=180)
                self.add_samples(self.patches_depth, self.patches_depth_next, patches_rotated, dual=False)

            elif k_index == 3:
                patches_flipped = self.flip(dual=True, side='vertical')
                patches_rotated = self.rotate(patches_flipped, dual=True, angle=270)
                self.add_samples(self.patches_depth_dual, self.patches_depth_next_dual, patches_rotated, dual=True)

            print(f'k index {k_index} is done')

    def delete_dry_patches(self):
        """
        Removes samples from the database where both depth and depth_next patches
        contain negligible water (sum < 2 meters combined across all 1024 cells).

        This helps exclude dry regions that are irrelevant for training.
        """
        indices_to_remove = []

        for i in range(len(self.database['depth'])):
            depth_sum = np.sum(self.database['depth'][i])
            depth_next_sum = np.sum(self.database['depth_next'][i])

            # If both patches have near-zero water (e.g., < 2m total), mark for deletion
            if (depth_sum + depth_next_sum) < 2:
                indices_to_remove.append(i)

        # Filter all keys in database using surviving indices
        for key in self.database:
            self.database[key] = [
                sample for i, sample in enumerate(self.database[key]) if i not in indices_to_remove]

    def generate_training_data(self):
        """
        Runs the full preprocessing pipeline:
        - Loads HDF and TIFF data
        - Computes grid size and trims water depth matrices
        - Extracts standard and dual terrain/depth patches
        - Applies flipping and rotation augmentations
        - Filters out dry (irrelevant) patches
        """
        self.populate_from_HDF()
        self.find_num_rows_cols_in_HECRAS()
        self.plot_depth_maps()
        self.load_tiff_data()
        self.calculate_num_patches()
        self.tiff_patches()
        self.populate_from_all_k_indices()
        self.delete_dry_patches()
