"""
This script extracts and processes flood simulation data from HDF files and TIFF terrain files
to generate patches from which datasets will be created, and then dataloader for deep learning models.
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
import torch
from scipy import ndimage
from config.paths_config import RAW_SIMULATIONS_DIR

# Set default tensor type to double precision for all PyTorch operations
torch.set_default_dtype(torch.float64)

class PatchExtractorProcessor:
    """
    PatchExtractorProcessor

    Processes raw flood simulation outputs (HDF and TIFF files) into structured
    patches suitable as deep learning samples. 

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
            prj_name (str): Project name as used in HECRAS (i set it when setting up the simulations)
            plan_num (str): Plan number for the simulation (HECRAS calls each simulation a plan)
        """
        # Project identifiers
        self.prj_num = prj_num
        self.prj_name = prj_name
        self.plan_num = plan_num

        # File paths (relative to GitHub root)
        self.prj_path = os.path.join(RAW_SIMULATIONS_DIR, f'prj_{prj_num}')
        self.terrain_path = os.path.join(self.prj_path, 'Terrains') # terrain are saved in simulations results directory
        self.plan_file_name = f'{prj_name}.p{plan_num}.hdf' # HECRAS saves simulations results with these names
        self.tiff_name = f'terrain_{plan_num}.tif'

        # Simulation settings
        self.k = 4  # Number of snapshots taken from each simulation
        self.delta_t = 60  # Time step between input and output in minutes. was set after to trial and error. 
        self.closest_indices = [0, 70, 140, 210]  # Time points used
        self.meters_in_cell = 10  # Cell resolution set in HECRAS
        self.cells_in_patch = 32  # Each sample is a 32x32 patch. was set after trial and error. 

        # Placeholders for internal data
        self.depth_vectors = None
        self.cells_center_coords = None
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
        self.trimmed_1km_inwards = False
        self.database = {'terrain': [], 'depth': [], 'depth_next': []} # end goal of using this class is to populate this dictionary.

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

            # Geometry: Cell center coordinates - convert to double precision
            data['Cells Center Coordinate'] = np.array(f['Geometry']['2D Flow Areas']['Perimeter 1']['Cells Center Coordinate'], dtype=np.float64)

            # Results: Invert depth values over time - convert to double precision
            results_group = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['Perimeter 1']
            data['Invert_Depth'] = np.array(results_group['Cell Invert Depth'], dtype=np.float64)

        return data

    def populate_from_HDF(self):
        """
        Populates depth vectors and cell center coordinates from the HDF file.
        """
        from_HDF_dict = self.from_HDF_file()

        # Depth vectors: 2D array [time, cell] - ensure double precision
        self.depth_vectors = from_HDF_dict['Invert_Depth'].astype(np.float64)

        # Cell center coordinates: 2D array [cell, (x, y)] - ensure double precision
        self.cells_center_coords = from_HDF_dict['Cells Center Coordinate'].astype(np.float64)


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
        k_depth_vectors = np.zeros((self.k, self.depth_vectors.shape[1]), dtype=np.float64)
        k_depth_vectors_next = np.zeros((self.k, self.depth_vectors.shape[1]), dtype=np.float64) # next is 60 mins ahead

        for i, idx in enumerate(self.closest_indices):
            k_depth_vectors[i] = self.depth_vectors[idx]
            k_depth_vectors_next[i] = self.depth_vectors[idx + self.delta_t]

        # Keep only valid cells (remove padding that HECRAS uses, called fictitious cells)
        k_depth_vectors = k_depth_vectors[:, :self.num_cells]
        k_depth_vectors_next = k_depth_vectors_next[:, :self.num_cells]

        new_shape = (self.k, self.num_rows, self.num_cols)

        # Reshape vectors into matrices (row-major order, same as HECRAS)
        self.k_depth_matrices = np.reshape(k_depth_vectors, new_shape).astype(np.float64)
        self.k_depth_matrices_next = np.reshape(k_depth_vectors_next, new_shape).astype(np.float64)

        # Optional trimming: Remove 1 km (100 cells) from each edge if domain is large to recude artificats found near boundaries 
        if self.num_rows > 240 and self.num_cols > 240: 
            self.trimmed_1km_inwards = True
            self.k_depth_matrices = self.k_depth_matrices[:, 100:-100, 100:-100]
            self.k_depth_matrices_next = self.k_depth_matrices_next[:, 100:-100, 100:-100]
            self.num_rows -= 200
            self.num_cols -= 200
            self.num_cells = self.num_rows * self.num_cols

    def load_tiff_data(self):
        """
        Loads the terrain TIFF file into self.tiff_data.
        If trimming is enabled, trims 1 km (1000 TIFF points) from each side.

        """
        tiff_path = os.path.join(self.terrain_path, self.tiff_name)
        self.tiff_data = tifffile.imread(tiff_path).astype(np.float64)  # Ensure double precision

        # Trim 1 km (1000 TIFF pixels) from edges if required
        if self.trimmed_1km_inwards:
            # Note: We trim only from the top and left to shift the origin of the TIFF.
            # This ensures alignment with the trimmed water depth grid, as terrain patches
            # will later be extracted according to the water depth cells.
            self.tiff_data = self.tiff_data[1000:, 1000:]
            
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
        self.patches_tiff = np.zeros((self.num_patches, self.tiff_points_in_patch, self.tiff_points_in_patch), dtype=np.float64)
        row_indices = np.arange(self.num_patches_row) * (self.tiff_points_in_patch - 1)
        col_indices = np.arange(self.num_patches_col) * (self.tiff_points_in_patch - 1)
        rows, cols = np.meshgrid(row_indices, col_indices)
        indices = np.column_stack((rows.ravel(), cols.ravel()))

        for i, (row, col) in enumerate(indices):
            patch = self.tiff_data[row:row + self.tiff_points_in_patch, col:col + self.tiff_points_in_patch]
            self.patches_tiff[i] = copy.deepcopy(patch)

        # --- Dual patches ---
        self.patches_tiff_dual = np.zeros((self.num_patches_dual, self.tiff_points_in_patch, self.tiff_points_in_patch), dtype=np.float64)

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
            self.patches_depth = np.zeros((self.num_patches, self.cells_in_patch, self.cells_in_patch), dtype=np.float64)
            self.patches_depth_next = np.zeros((self.num_patches, self.cells_in_patch, self.cells_in_patch), dtype=np.float64)

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
            self.patches_depth_dual = np.zeros((self.num_patches_dual, self.cells_in_patch, self.cells_in_patch), dtype=np.float64)
            self.patches_depth_next_dual = np.zeros((self.num_patches_dual, self.cells_in_patch, self.cells_in_patch), dtype=np.float64)

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

        for patch_depth, patch_depth_next, patch_terrain in zip(patches_depth, patches_depth_next, patches_tiff):
            sample = {
                'terrain': patch_terrain.astype(np.float64),  # Ensure double precision
                'depth': patch_depth.astype(np.float64),      # Ensure double precision
                'depth_next': patch_depth_next.astype(np.float64)  # Ensure double precision
            }
            for key, value in sample.items():
                self.database[key].append(value)


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
        for k_index in [0, 1, 2, 3]:
            self.depth_patches(k_index=k_index) #It extracts a fresh set of patches (depth, depth_next, and optionally dual versions) for the specific k_index.

            if k_index == 0:
                self.add_samples(self.patches_depth, self.patches_depth_next, self.patches_tiff, dual=False)

            elif k_index == 1:
                patches_tiff_flipped = self.flip(dual=True, side='horizontal')
                patches_tiff_rotated = self.rotate(patches_tiff_flipped, dual=True, angle=90)
                self.add_samples(self.patches_depth_dual, self.patches_depth_next_dual, patches_tiff_rotated, dual=True)

            elif k_index == 2:
                patches_tiff_flipped = self.flip(dual=False, side='vertical')
                patches_tiff_rotated = self.rotate(patches_tiff_flipped, dual=False, angle=180)
                self.add_samples(self.patches_depth, self.patches_depth_next, patches_tiff_rotated, dual=False)

            elif k_index == 3:
                patches_tiff_flipped = self.flip(dual=True, side='vertical')
                self.add_samples(self.patches_depth_dual, self.patches_depth_next_dual, patches_tiff_flipped, dual=True)

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
            self.database[key] = [sample for i, sample in enumerate(self.database[key]) if i not in indices_to_remove]

    def postprocess_patches(self):
        """
        Postprocesses patches by:
        - Removing samples where depth_next - depth contains large values (>|10| m)
        - Shifting each terrain patch so its minimum value is zero
        """
        keep_indices = []
    
        for i in range(len(self.database['depth'])):
            diff = self.database['depth_next'][i] - self.database['depth'][i]
            if np.max(np.abs(diff)) <= 10:
                keep_indices.append(i)
    
        # Apply filtering
        for key in self.database:
            self.database[key] = [self.database[key][i] for i in keep_indices]
    
        # Shift terrain patches - ensure double precision is maintained
        self.database['terrain'] = [(terrain - np.min(terrain)).astype(np.float64) for terrain in self.database['terrain']]

    def save_patches(self):
        """
        Saves the patches to pickle files as expected by generate_datasets.py.
        Creates files:
        - prj_X_plan_Y_terrain_patches.pkl
        - prj_X_plan_Y_depth_patches.pkl  
        - prj_X_plan_Y_depth_next_patches.pkl
        """
        # Import here to avoid circular imports
        from config.paths_config import PATCHES_DIR
        
        # Create project-specific directory
        patches_dir = PATCHES_DIR/f"prj_{self.prj_num}/plan_{self.plan_num}"
        patches_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we have patches to save
        if not self.database['depth']:
            print(f"No patches to save for prj_{self.prj_num} plan_{self.plan_num}")
            return
        
        # Debug: Check actual dtypes before saving
        print(f"Saving patches for prj_{self.prj_num} plan_{self.plan_num}:")
        for key in ['terrain', 'depth', 'depth_next']:
            if self.database[key]:
                sample_dtype = self.database[key][0].dtype
                print(f"  {key}: {sample_dtype} (shape: {self.database[key][0].shape})")
        
        # Save each patch type to separate pickle files
        for key in ['terrain', 'depth', 'depth_next']:
            if self.database[key]:
                filename = f'{key}_patches.pkl'
                filepath = patches_dir / filename
                
                with open(filepath, 'wb') as f:
                    pickle.dump(self.database[key], f)
                        
    def plot_final_patches(self):
        # Unpack the data
        depths = self.database['depth']
        depths_next = self.database['depth_next']
        terrains = self.database['terrain']
    
        n_total = len(depths)
        n_samples = min(len(depths),10)  # don't exceed available samples
        idxs = random.sample(range(n_total), n_samples)
    
        # Create output directory
        output_dir = os.path.join(
            'simulations_to_samples',
            'processed_data',
            'patches_per_simulation',
            'figures')
        os.makedirs(output_dir, exist_ok=True)
    
        # Create figure: 10 rows x 3 columns
        fig, axs = plt.subplots(nrows=10, ncols=3, figsize=(12, 30))
        for i, idx in enumerate(idxs): 
            ax = axs[i, 0]
            ax.imshow(depths[idx], cmap='viridis')
            ax.set_title(f'Depth [{idx}]')
            
            ax = axs[i, 1]
            ax.imshow(depths_next[idx], cmap='viridis')
            ax.set_title(f'Depth Next [{idx}]')

            ax = axs[i, 2]
            ax.imshow(terrains[idx], cmap='terrain')
            ax.set_title(f'Terrain [{idx}]')

            ax.axis('off')
    
        plt.tight_layout()
    
        # Save single image for this simulation (plan)
        save_path = os.path.join(output_dir, f'prj_{self.prj_num}_plan_{self.plan_num}_random_patches.png')
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        
    def plot_and_save_maps(self):
        """
        Plots the original terrain TIFF, 4 depth maps, 4 depth_next maps,
        and their differences for the current plan, then saves the figure in the raw_data/images/ directory.
        """
        # Prepare the figure: 4 rows, 3 columns (for terrain, depth, depth_next, and difference maps)
        fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(40, 20))
        # Plot Terrain (first row, first column)
        axs[0, 0].imshow(self.tiff_data, cmap='terrain')
        axs[0, 0].set_title('Original Terrain')
        axs[0, 0].axis('off')
        cbar = plt.colorbar(axs[0, 0].imshow(self.tiff_data, cmap='terrain'), ax=axs[0, 0])
        cbar.set_label('Meters')
        # Plot Depth (Depth 0–3 in columns 1–4, row 2)
        for i in range(4):
            #print(self.k_depth_matrices.shape)
            axs[1, i].imshow(self.k_depth_matrices[i], cmap='Blues')
            axs[1, i].set_title(f'Depth {i}')
            axs[1, i].axis('off')
            cbar = plt.colorbar(axs[1, i].imshow(self.k_depth_matrices[i], cmap='Blues'), ax=axs[1, i])
            cbar.set_label('Meters')
        # Plot Depth Next (Depth Next 0–3 in columns 1–4, row 3)
        for i in range(4):
            axs[2, i].imshow(self.k_depth_matrices_next[i], cmap='Blues')
            axs[2, i].set_title(f'Depth Next {i}')
            axs[2, i].axis('off')
            cbar = plt.colorbar(axs[2, i].imshow(self.k_depth_matrices_next[i], cmap='Blues'), ax=axs[2, i])
            cbar.set_label('Meters')
        # Plot Difference (Depth Next - Depth) in row 4
        for i in range(4):
            diff = self.k_depth_matrices_next[i] - self.k_depth_matrices[i]
            axs[3, i].imshow(diff, cmap='coolwarm')
            axs[3, i].set_title(f'Diff {i}')
            axs[3, i].axis('off')
            cbar = plt.colorbar(axs[3, i].imshow(diff, cmap='coolwarm'), ax=axs[3, i])
            cbar.set_label('Meters')
        # Remove unnecessary axis for row 4 (empty column 0)
        axs[0, 1].axis('off')
        axs[0, 2].axis('off')
        axs[0, 3].axis('off')

        # Save the figure in the raw_data/images/ directory
        output_dir = os.path.join(RAW_SIMULATIONS_DIR, 'figures')
        os.makedirs(output_dir, exist_ok=True)
    
        # Save the figure as PNG with the project and plan identifiers
        save_path = os.path.join(output_dir, f'prj_{self.prj_num}_plan_{self.plan_num}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

    def generate_patches(self):
        """
        Runs the full preprocessing pipeline:
        - Loads HDF and TIFF data
        - Computes grid size and trims water depth matrices
        - Extracts standard and dual terrain/depth patches
        - Applies flipping and rotation augmentations
        - Filters out dry (irrelevant) and unstable patches, postprocesses terrain
        """
        self.populate_from_HDF()
        self.find_num_rows_cols_in_HECRAS()
        self.load_tiff_data()
        self.calculate_num_patches()
        self.tiff_patches()
        self.populate_from_all_k_indices()
        self.delete_dry_patches()
        self.postprocess_patches()
        
        # Only plot if self.plot is True
        if self.plot:
            self.plot_and_save_maps()
            self.plot_final_patches()
        
        # Save patches to pickle files
        self.save_patches()
