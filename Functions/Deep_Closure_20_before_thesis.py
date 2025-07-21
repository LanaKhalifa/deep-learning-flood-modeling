#%% Import Libraries
import h5py
import numpy as np
import os
import tifffile
import copy
import torch
import matplotlib.pyplot as plt

#%% Deep_Closure class
class Deep_Closure:
    # CORRECT
    def __init__(self,
                 prj_num='02',
                 prj_name='HECRAS_on_02', # must be check it for each project
                 plan_num='02',
                 cells_in_patch=32,
                 t = 208,
                 tolerance = 10e-10,
                 delta_t = 60): # must iterate over plan number. create a list for all plan numbers.

        # HDF file name and location
        self.prj_num = prj_num
        self.prj_name = prj_name
        self.plan_num = plan_num

        self.prj_path = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/HECRAS_Simulations_Results/prj_' + self.prj_num 
        self.terrain_path = self.prj_path + '/Terrains'

        self.plan_file_name = prj_name + '.p' + self.plan_num + '.hdf'
        self.tiff_name = 'terrain_'+plan_num+'.tif'

        self.meters_in_cell = 10 # even number and must stay even 
        
        if cells_in_patch % 2 == 0:
            self.cells_in_patch = cells_in_patch
        else: 
            raise ValueError("Error: cells_in_patch must be even to allow creating dual patches.")

        
        self.t = t # time in which closure model will be applied between t and t+60
        self.delta_t = 60
        self.tolerance = tolerance
        
        self.patches_depth_next_dict = {'A': None,
                                        'B': None,
                                        'C': None,
                                        'D': None}
        
        self.patches_depth_dict = {'A': None,
                                   'B': None,
                                   'C': None,
                                   'D': None}
        
        self.patches_true_depth_next_dict = {'A': None,
                                             'B': None,
                                             'C': None,
                                             'D': None}
        
        self.num_patches_dict = {'A': None,
                                'B': None,
                                'C': None,
                                'D': None}
        
        self.num_patches_row_dict = {'A': None,
                                    'B': None,
                                    'C': None,
                                    'D': None}  
        
        self.num_patches_col_dict = {'A': None,
                                    'B': None,
                                    'C': None,
                                    'D': None} 
        
        self.patches_tiff = {'A': None,
                             'B': None,
                             'C': None,
                             'D': None} 
        self.saved_initial_BD = None
        
    def from_HDF_file(self): 
        with h5py.File(self.plan_file_name,'r') as f:
            
            from_HDF = {}
            Results = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['Perimeter 1']
            from_HDF['Invert_Depth'] = np.array(Results['Cell Invert Depth'], dtype=np.float64)
            from_HDF['Cells Center Coordinate'] = np.array(f['Geometry']['2D Flow Areas']['Perimeter 1']['Cells Center Coordinate'], dtype=np.float64)
            
            return from_HDF
    def populate_from_HDF(self): 
        os.chdir(self.prj_path)
           
        from_HDF_dict = self.from_HDF_file()
        self.depth_vectors = from_HDF_dict['Invert_Depth'] # 2D: rows = time dimension. cols = cells
        self.cells_center_coords = from_HDF_dict['Cells Center Coordinate'] # 2D: rows = cells. cols = x, y. 

        valid_time_indices = []
        
        # Loop through the time dimension (excluding the last time step)
        for t in range(len(self.depth_vectors) - (self.delta_t)):
            # Calculate the difference between consecutive time steps
            diff = self.depth_vectors[t] - self.depth_vectors[t + self.delta_t]
            
            # Check if the maximum difference is less than 1
            if np.max(np.abs(diff)) < 1:
                # Store the index
                valid_time_indices.append(t)
                
        if self.t not in valid_time_indices:
            print('ERROR: Please change t')
            print("Valid time indices:", valid_time_indices)
            
    def find_num_rows_cols_in_HECRAS(self): 
        threshold = 1  # Define a threshold for significant difference
        # Count the number of rows where the second entry is close to cells_center_coords[0][1]
        count = sum(1 for row in self.cells_center_coords if abs(row[1] - self.cells_center_coords[0][1]) < threshold)
        self.num_cols = count - 2 # HECRAS used outer cells 
        # Count the number of rows where the second entry is close to cells_center_coords[0][0]
        count = sum(1 for row in self.cells_center_coords if abs(row[0] - self.cells_center_coords[0][0]) < threshold)
        self.num_rows = count - 2 # HECRAS used outer cells 

        self.num_cells = self.num_rows * self.num_cols
    def one_matrix_depth_WSE(self): 
        depth_vector = self.depth_vectors[self.t]
        depth_vector_next = self.depth_vectors[self.t+self.delta_t]
        
        depth_vector = depth_vector[ :self.num_cells]
        depth_vector_next = depth_vector_next[:self.num_cells]   
        
        new_shape = (self.num_rows, self.num_cols) 
        
        #np.reshape works exactly like HECRAS stores cells (row major manner)
        self.depth_matrix = np.reshape(depth_vector, new_shape) # (num_rows, num_cols)
        self.depth_matrix_next = np.reshape(depth_vector_next, new_shape) # ( num_rows, num_cols)
        
        # Plot self.depth_matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(self.depth_matrix, cmap='Blues')
        plt.colorbar(orientation='vertical')
        plt.title('1. Depth Matrix - from HECRAS')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()
        
        # Plot self.depth_matrix_next
        plt.figure(figsize=(10, 8))
        plt.imshow(self.depth_matrix_next, cmap='Blues')
        plt.colorbar(orientation='vertical')
        plt.title('2. Depth Matrix Next - from HECRAS')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()
        
        # Calculate the difference matrix
        difference_matrix = self.depth_matrix_next - self.depth_matrix
        
        
        # Plot self.depth_matrix_next
        plt.figure(figsize=(10, 8))
        plt.imshow(difference_matrix, cmap='Blues')
        plt.colorbar(orientation='vertical')
        plt.title('3. Difference Matrix - from HECRAS')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()        
        
        
        # Create a figure with three subplots
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        
        # Set the title for the entire figure
        fig.suptitle('4. Matrix, Matrix next, and Matrix Difference - from HECRAS', fontsize=30)
        
        # Plot self.depth_matrix
        im1 = axes[0].imshow(self.depth_matrix, cmap='Blues')
        axes[0].set_title('Depth Matrix')
        fig.colorbar(im1, ax=axes[0], orientation='vertical')
        
        # Plot self.depth_matrix_next
        im2 = axes[1].imshow(self.depth_matrix_next, cmap='Blues')
        axes[1].set_title('Depth Matrix Next')
        fig.colorbar(im2, ax=axes[1], orientation='vertical')
        
        # Plot the difference matrix
        im3 = axes[2].imshow(difference_matrix, cmap='Blues')
        axes[2].set_title('Difference (Next - Current)')
        fig.colorbar(im3, ax=axes[2], orientation='vertical')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the suptitle
        
        # Display the plots
        plt.show()        
    def load_tiff_data(self): 
        os.chdir(self.terrain_path)
        self.tiff_data = tifffile.imread(self.tiff_name).astype(np.float64)
        
        # Plot the tiff_data
        plt.figure(figsize=(10, 8))
        plt.imshow(self.tiff_data, cmap='Blues')
        plt.colorbar(orientation='vertical')
        plt.title('5. TIFF Data')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()        
    def calculate_num_patches(self): 
        # Calculate patch size
        self.meters_in_patch = self.meters_in_cell * self.cells_in_patch # both are even, so meters_in_patch is even
        self.tiff_points_in_patch = self.meters_in_patch + 1  # e.g., for 2m terrain, you need 3 points (edges and middle)
        
        # Calculate num of patches
        self.num_patches_row_dict['A'] = (self.num_rows - 1) // self.cells_in_patch  # num patches in row dimension
        self.num_patches_col_dict['A'] = (self.num_cols - 1) // self.cells_in_patch # num patches in col dimension
        self.num_patches_dict['A'] = self.num_patches_row_dict['A'] * self.num_patches_col_dict['A']
        
        # Update number of rows and cols in HECRAS since we have eliminated some
        self.num_rows = self.num_patches_row_dict['A'] * self.cells_in_patch
        self.num_cols = self.num_patches_col_dict['A'] * self.cells_in_patch
        
        # cut tiff data such that it encompasses only our patches
        num_tiff_points_in_all_patches_row = self.num_patches_row_dict['A'] * self.cells_in_patch * self.meters_in_cell + 1 
        num_tiff_points_in_all_patches_col = self.num_patches_col_dict['A'] * self.cells_in_patch * self.meters_in_cell + 1 
        self.tiff_data = self.tiff_data[:num_tiff_points_in_all_patches_row, :num_tiff_points_in_all_patches_col]
        
        # Plot the tiff_data after clipping around A
        plt.figure(figsize=(10, 8))
        plt.imshow(self.tiff_data, cmap='Blues')
        plt.colorbar(orientation='vertical')
        plt.title('6. TIFF Data after clipping')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()
        
        self.depth_matrix = self.depth_matrix[:self.num_rows, :self.num_cols]
        self.depth_matrix_next = self.depth_matrix_next[:self.num_rows, :self.num_cols]
        
        # Plot self.depth_matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(self.depth_matrix, cmap='Blues')
        plt.colorbar(orientation='vertical')
        plt.title('7. Depth Matrix - from HECRAS - after clipping - with A Boundaries')
        plt.xlabel('Columns')
        plt.ylabel('Rows')

        # Get the dimensions of the matrix
        num_rows, num_cols = self.depth_matrix.shape

        # Draw the boundaries of each 10x10 patch
        for row in range(0, num_rows, self.cells_in_patch):
            plt.axhline(row - 0.5, color='red', linewidth=1.5)  # Horizontal line
        for col in range(0, num_cols, self.cells_in_patch):
            plt.axvline(col - 0.5, color='red', linewidth=1.5)  # Vertical line
        plt.show()
        
        # Plot self.depth_matrix_next
        plt.figure(figsize=(10, 8))
        plt.imshow(self.depth_matrix_next, cmap='Blues')
        plt.colorbar(orientation='vertical')
        plt.title('8. Depth Matrix Next - from HECRAS - after clipping')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()
                
        
        self.saved_true_matrix_next = copy.deepcopy(self.depth_matrix_next)
        self.depth_matrix_next = self.previous_depth_internal(self.depth_matrix_next) # for creating input 
        self.depth_matrix_next_dummy= copy.deepcopy(self.depth_matrix_next) 
        
        # when looking at dual patches we throw the first cells_in_patch/2 in both dims. from there the same computation should be performed. as the last cell must be discarded in both dims to avoid non square cells.
        self.num_patches_row_dict['B'] = self.num_patches_row_dict['A'] - 1
        self.num_patches_col_dict['B'] = self.num_patches_col_dict['A'] - 1
        self.num_patches_dict['B'] = self.num_patches_row_dict['B'] * self.num_patches_col_dict['B']

        self.num_patches_row_dict['C'] = self.num_patches_row_dict['A'] 
        self.num_patches_col_dict['C'] = self.num_patches_col_dict['A'] - 1
        self.num_patches_dict['C'] = self.num_patches_row_dict['C'] * self.num_patches_col_dict['C']

        self.num_patches_row_dict['D'] = self.num_patches_row_dict['A'] - 1
        self.num_patches_col_dict['D'] = self.num_patches_col_dict['A'] 
        self.num_patches_dict['D'] = self.num_patches_row_dict['D'] * self.num_patches_col_dict['D']

    # EXTRACTING PATCHES
    def tiff_patches(self, trimmed_tiff_data, num_patches_row, num_patches_col, num_patches): 
        patches_tiff = np.zeros((num_patches, self.tiff_points_in_patch, self.tiff_points_in_patch), dtype=np.float64)
        row_indices = np.arange(0, num_patches_row, 1)*(self.tiff_points_in_patch - 1) # on a paper, think, suppose each patch has 5 tiff points, then the indices would be [0-5, 5-10, 10-15, 15-20]. so, the indices are [0,5,15,..]. 5 is tiff_points_in_patch - 1
        col_indices =  np.arange(0, num_patches_col , 1)*(self.tiff_points_in_patch - 1) # just like columns
        rows, cols = np.meshgrid(row_indices, col_indices)
        # ravel() returns a 1-D array  of the input
        # indices for the upper left corner of each tiff patch are returned. we startd from the upper left corner of the big tiff, and return the upper left corners of the patches of the 1st column patches, then the 2nd column patches...
        indices = np.column_stack((rows.ravel(), cols.ravel())) # it provides the indices of patches in a column major manner: [0,0], [10,0], [20,0]
        
        trimmed = copy.deepcopy(trimmed_tiff_data)
        for i, (row, col) in enumerate(indices):
          patch = trimmed[row : row+self.tiff_points_in_patch, col : col+self.tiff_points_in_patch] # row:row+tiff_points_in_patch 0:11 is 0,1,2,3,4,5,7,8,9,10 (exactly as needed)
          patches_tiff[i] = patch - np.min(patch)
         

# =============================================================================
#         # Determine the global minimum and maximum values across all patches
#         vmin = np.min([np.min(patch) for patch in patches_tiff])
#         vmax = np.max([np.max(patch) for patch in patches_tiff])
#         # Create a figure with subplots
#         fig, axes = plt.subplots(num_patches_row, num_patches_col, figsize=(num_patches_col * 3, num_patches_row * 3))
#         for j in range(num_patches_col):
#             for i in range(num_patches_row):
#                 idx = j * num_patches_row + i  # Change the indexing for column-major order
#                 ax = axes[i, j]
#                 im = ax.imshow(patches_tiff[idx], cmap='Blues', vmin=vmin, vmax=vmax)
#                 ax.set_title(f'Patch {idx + 1}')
#                 ax.axis('off')
#         # Add a colorbar
#         cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
#         # Set the title for the entire figure
#         fig.suptitle('tiff patches', fontsize=50)
#         # Display the plots
#         plt.show()     
# =============================================================================
                 
            
        return patches_tiff
    def populate_tiff_patches(self): 
        #this will be run once and untouched since tiff patches do not change 
        which = 'A'
        self.patches_tiff[which] =  self.tiff_patches(trimmed_tiff_data = self.tiff_data, # no trimming for type A. 
                                                        num_patches_row = self.num_patches_row_dict[which],
                                                        num_patches_col = self.num_patches_col_dict[which],
                                                        num_patches = self.num_patches_dict[which])
    
        num_patches_row = self.num_patches_row_dict[which]
        num_patches_col = self.num_patches_col_dict[which]
        patches = self.patches_tiff[which]
               
        strts_from = int(self.meters_in_patch/2) #throw tiff points corresponding to the first 5 cells only. which are 0-49. the tiff point at the location of 50, is for the sixth cell.  

        which = 'B'
        self.patches_tiff[which] = self.tiff_patches(trimmed_tiff_data = self.tiff_data[strts_from:-strts_from, strts_from:-strts_from],
                                                     num_patches_row = self.num_patches_row_dict[which],
                                                     num_patches_col = self.num_patches_col_dict[which],
                                                     num_patches = self.num_patches_dict[which])

        
        num_patches_row = self.num_patches_row_dict[which]
        num_patches_col = self.num_patches_col_dict[which]
        patches = self.patches_tiff[which]
           
        
        which = 'C'
        self.patches_tiff[which] = self.tiff_patches(trimmed_tiff_data = self.tiff_data[:, strts_from:-strts_from], #must doubles check the trimming
                                                     num_patches_row = self.num_patches_row_dict[which],
                                                     num_patches_col = self.num_patches_col_dict[which],
                                                     num_patches = self.num_patches_dict[which])
        
        num_patches_row = self.num_patches_row_dict[which]
        num_patches_col = self.num_patches_col_dict[which]
        patches = self.patches_tiff[which]


        which = 'D'
        self.patches_tiff[which] = self.tiff_patches(trimmed_tiff_data = self.tiff_data[strts_from:-strts_from, :], #must doubles check the trimming
                                                    num_patches_row = self.num_patches_row_dict[which],
                                                    num_patches_col = self.num_patches_col_dict[which],
                                                    num_patches = self.num_patches_dict[which])    
    def var_patches(self, trimmmed_var_data, num_patches_row, num_patches_col, num_patches):
        patches_var = np.zeros((num_patches, self.cells_in_patch, self.cells_in_patch), dtype=np.float64)

        # just like from from_HDF
        row_indices = np.arange(0, num_patches_row, 1) * self.cells_in_patch # on a paper think, suppose each patch holds 10 cells, then the indices would be [0-9, 10-19, 20,29..]. so the indices are [0, 10, 20]. 10 is cells_in_patch
        col_indices =  np.arange(0, num_patches_col, 1) * self.cells_in_patch
        rows, cols = np.meshgrid(row_indices, col_indices)
        indices = np.column_stack((rows.ravel(), cols.ravel()))

        for i, (row, col) in enumerate(indices):
            patch = trimmmed_var_data[row:row+self.cells_in_patch, col:col+self.cells_in_patch] 
            patches_var[i] =  copy.deepcopy(patch)
           
# =============================================================================
#         if plot:
#             # Determine the global minimum and maximum values across all patches
#             vmin = np.min([np.min(patch) for patch in patches_var])
#             vmax = np.max([np.max(patch) for patch in patches_var])
#             # Create a figure with subplots
#             fig, axes = plt.subplots(num_patches_row, num_patches_col, figsize=(num_patches_col * 3, num_patches_row * 3))
#             for j in range(num_patches_col):
#                 for i in range(num_patches_row):
#                     idx = j * num_patches_row + i  # Change the indexing for column-major order
#                     ax = axes[i, j]
#                     im = ax.imshow(patches_var[idx], cmap='Blues', vmin=vmin, vmax=vmax)
#                     ax.set_title(f'Patch {idx + 1}')
#                     ax.axis('off')
#             # Add a colorbar
#             cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
#             # Set the title for the entire figure
#             fig.suptitle('Depth patches', fontsize=20)
#             # Display the plots
#             plt.show()     
#         
# =============================================================================
        return patches_var
    def populate_depth_patches(self): 
        # this function will be run only one, since depth matrix is not update at n+1
        depth_matrix = copy.deepcopy(self.depth_matrix)
        which = 'A'
        self.patches_depth_dict[which] =  self.var_patches(
                                                              trimmmed_var_data = depth_matrix,
                                                              num_patches_row = self.num_patches_row_dict[which],
                                                              num_patches_col = self.num_patches_col_dict[which],
                                                              num_patches = self.num_patches_dict[which])
            
        strts_from = int(self.cells_in_patch/2)

        num_patches_row = self.num_patches_row_dict[which]
        num_patches_col = self.num_patches_col_dict[which]
        patches = self.patches_depth_dict[which]
        
        
        which = 'B'
        self.patches_depth_dict[which] = self.var_patches(
                                                            trimmmed_var_data = depth_matrix[strts_from:-strts_from, strts_from:-strts_from], #must doubles check the trimming
                                                            num_patches_row = self.num_patches_row_dict[which],
                                                            num_patches_col = self.num_patches_col_dict[which],
                                                            num_patches = self.num_patches_dict[which])

        num_patches_row = self.num_patches_row_dict[which]
        num_patches_col = self.num_patches_col_dict[which]
        patches = self.patches_depth_dict[which]
        

        which = 'C'
        self.patches_depth_dict[which]= self.var_patches(
                                                            trimmmed_var_data = depth_matrix[:, strts_from:-strts_from], #must doubles check the trimming
                                                            num_patches_row = self.num_patches_row_dict[which],
                                                            num_patches_col = self.num_patches_col_dict[which],
                                                            num_patches = self.num_patches_dict[which])

        num_patches_row = self.num_patches_row_dict[which]
        num_patches_col = self.num_patches_col_dict[which]
        patches = self.patches_depth_dict[which]
        
        which = 'D'
        self.patches_depth_dict[which] = self.var_patches(
                                                            trimmmed_var_data = depth_matrix[strts_from:-strts_from, :], #must doubles check the trimming
                                                            num_patches_row = self.num_patches_row_dict[which],
                                                            num_patches_col = self.num_patches_col_dict[which],
                                                            num_patches = self.num_patches_dict[which])

        num_patches_row = self.num_patches_row_dict[which]
        num_patches_col = self.num_patches_col_dict[which]
        patches = self.patches_depth_dict[which]     
    def populate_depth_next_patches(self, which): 
        depth_matrix_next = copy.deepcopy(self.depth_matrix_next)    
        if which == 'A':
            self.patches_depth_next_dict[which] =  self.var_patches(
                                                                    trimmmed_var_data = depth_matrix_next,
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])
                        
            num_patches_row = self.num_patches_row_dict[which]
            num_patches_col = self.num_patches_col_dict[which]
            patches = self.patches_depth_next_dict[which]
            

        strts_from = int(self.cells_in_patch/2)
        
        if which == 'B':
            self.patches_depth_next_dict[which] = self.var_patches(
                                                                    trimmmed_var_data = depth_matrix_next[strts_from:-strts_from, strts_from:-strts_from], # removes first and last strts_from rows and cols
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])

            num_patches_row = self.num_patches_row_dict[which]
            num_patches_col = self.num_patches_col_dict[which]
            patches = self.patches_depth_next_dict[which]
            
        if which == 'C':
            self.patches_depth_next_dict[which] = self.var_patches(
                                                                    trimmmed_var_data = depth_matrix_next[:, strts_from:-strts_from], #must doubles check the trimming
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])
                
            num_patches_row = self.num_patches_row_dict[which]
            num_patches_col = self.num_patches_col_dict[which]
            patches = self.patches_depth_next_dict[which]
            
            
        if which == 'D':
            self.patches_depth_next_dict[which] = self.var_patches(
                                                                    trimmmed_var_data = depth_matrix_next[strts_from:-strts_from, :], #must doubles check the trimming
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])
            num_patches_row = self.num_patches_row_dict[which]
            num_patches_col = self.num_patches_col_dict[which]
            patches = self.patches_depth_next_dict[which]         
    def populate_true_depth_next_patches(self, which): 
        depth_matrix_next = copy.deepcopy(self.saved_true_matrix_next)    
        if which == 'A':
            self.patches_true_depth_next_dict[which] =  self.var_patches(
                                                                    trimmmed_var_data = depth_matrix_next,
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])
                        
        strts_from = int(self.cells_in_patch/2)
        
        if which == 'B':
            self.patches_true_depth_next_dict[which] = self.var_patches(
                                                                    trimmmed_var_data = depth_matrix_next[strts_from:-strts_from, strts_from:-strts_from], # removes first and last strts_from rows and cols
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])
                    
        if which == 'C':
            self.patches_true_depth_next_dict[which] = self.var_patches(
                                                                    trimmmed_var_data = depth_matrix_next[:, strts_from:-strts_from], #must doubles check the trimming
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])
                
        if which == 'D':
            self.patches_true_depth_next_dict[which] = self.var_patches(
                                                                    trimmmed_var_data = depth_matrix_next[strts_from:-strts_from, :], #must doubles check the trimming
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])
      
    # HELPING FUNCTIONS                                   
    def zero_internal(self, matrix):
        matrix_to_change =  copy.deepcopy(matrix)
        rows, cols = matrix_to_change.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                matrix_to_change[i][j] = 0
        return matrix_to_change
    def zero_BC(self, matrix):
        rows, cols = matrix.shape
        matrix_to_change = copy.deepcopy(matrix)
        for i in range(0, rows):
                matrix_to_change[i][0] = 0
                matrix_to_change[i][-1] = 0
                
        for j in range(0, cols):
                matrix_to_change[0][j] = 0
                matrix_to_change[-1][j] = 0                
        return matrix_to_change
    def previous_depth_internal(self, matrix):
        matrix_to_change = copy.deepcopy(matrix)
        depth_matrix = copy.deepcopy(self.depth_matrix)
        rows, cols = matrix_to_change.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                matrix_to_change[i][j] = depth_matrix[i][j]
        return matrix_to_change
     
    # CREATE TENSORS, LOAD TRAINED MODEL AND FORWARD PASS
    def depth_tensors_model_input(self, which):        
        # Get the relevant attributes based on 'which'
        patches_depth_next = copy.deepcopy(self.patches_depth_next_dict[which]) # self.patches_depth_next_dict[which] is (num_patches[which], cells_in_patch, cells_in_patch)
        patches_depth_next_BC = [self.zero_internal(matrix) for matrix in patches_depth_next]
        
        patches_depth = copy.deepcopy(self.patches_depth_dict[which])
        
        # Initialize an empty list to store concatenated patches
        patches_data = []
        num_patches = self.num_patches_dict[which]
        
        for i in range(num_patches):
            concatenated_patch = np.stack((patches_depth_next_BC[i], patches_depth[i]), axis=0) # stacking as opposed to concatenation adds a new dim
            patches_data.append(concatenated_patch)
    
        patches_data = np.array(patches_data, dtype=np.float64) # convert list into an array
    
        return torch.tensor(patches_data, dtype=torch.double).cuda() # torch tensor (N, 2, cells_in_patch, cells_in_patch)
    def tiff_tensors_model_input(self, which):
        tiff_patches = copy.deepcopy(self.patches_tiff[which])        
        tiff_patches = np.expand_dims(tiff_patches, axis=1)
        
        return torch.tensor(tiff_patches, dtype=torch.double).cuda() # (N, 1, tiff_point_in_patch, tiff_point_in_patch)
    def load_trained_model(self):
        shared_terrain_downsample = TerrainDownsampleModel(c_start=10, c_end=1, act='leakyrelu')           
        shared_terrain_downsample = TerrainDownsampleModel(c_start=10, c_end=1, act='leakyrelu').cuda()
        self.model = SimpleFlexibleCNN(downsampler=shared_terrain_downsample, num_layers=6, num_channels=32, input_channels=3, act='leakyrelu').cuda()
        # Load the model's state dictionary
        folder = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Architecture_04/Architecture_04_reduce_runtime_double/trained_models/'
        model_name = '0_model'
        path = folder + model_name + '.pth'
        self.model.load_state_dict(torch.load(path))
        # Set the model to evaluation mode (if you plan to use it for inference)
        print("Model loaded and ready for inference") 
    def forward_pass(self, which, debug=False):  # MUST WORK ON IT TONIGHT
        if not debug:
            # Create tensors from depth_an_BC and terrain
            depth_input = self.depth_tensors_model_input(which)
            terrain_input = self.tiff_tensors_model_input(which)
           
            self.model.eval()
            outputs = self.model(terrain_input, depth_input) # output is a tensor (N, 1, num_cells, num_cells)
            outputs = outputs[:,0,:,:]
            # Assuming outputs is a PyTorch tensor of dimensions (N, 1, num_cells, num_cells)
            outputs_squeezed = outputs.squeeze(dim=1)  # remove the dimension at dim=1
            outputs_numpy = outputs_squeezed.cpu().detach().numpy()   # convert the tensor to a NumPy array
            
            self.patches_depth_next_dict[which] = outputs_numpy + self.patches_depth_dict[which]    
        
        if debug:
            self.populate_true_depth_next_patches(which)
            patches = copy.deepcopy(self.patches_true_depth_next_dict[which])
            noisy_patches = [patch + np.random.normal(0, 0.1, patch.shape) for patch in patches]
            self.patches_depth_next_dict[which] = noisy_patches
            
            
    # RECONSTRUCT MATRIX
    def reconstruct_depth_matrix_next(self, which): # chatGPT write it, must check it. 
        patches = self.patches_depth_next_dict[which] # this assumes patches_depth_next_dict is udpated after forward pass
        num_patches_row = self.num_patches_row_dict[which]
        num_patches_col = self.num_patches_col_dict[which]

        # Create an empty array for the reconstructed depth matrix
        depth_next_matrix = np.zeros((num_patches_row * self.cells_in_patch, num_patches_col * self.cells_in_patch), dtype=np.float64)

        # populate this empty matrix with the output
        for i in range(num_patches_col): 
            for j in range(num_patches_row):
                patch_index = i * num_patches_row + j
                row_start = j * self.cells_in_patch
                row_end = (j + 1) * self.cells_in_patch
                col_start = i * self.cells_in_patch
                col_end = (i + 1) * self.cells_in_patch
                depth_next_matrix[row_start:row_end, col_start:col_end] = patches[patch_index]
        

        
        strts_from = int(self.cells_in_patch/2)
        if which == 'A':
            self.depth_matrix_next[1:-1, 1:-1] = depth_next_matrix[1:-1, 1:-1] # you want to avoid touching real BC
            
        if which == 'B':
            self.depth_matrix_next[strts_from:-strts_from, strts_from:-strts_from] = depth_next_matrix
            
        if which == 'C':
            self.depth_matrix_next[1:-1, strts_from:-strts_from] =  depth_next_matrix[1:-1, :]
            
        if which == 'D':
            self.depth_matrix_next[strts_from:-strts_from, 1:-1] = depth_next_matrix[:, 1:-1]
          
# =============================================================================
#         plt.figure(figsize=(10, 8))
#         plt.imshow(self.depth_matrix_next, cmap='Blues')
#         plt.colorbar(orientation='vertical')
#         plt.title(f'Reconstructed Depth Next Matrix in (self.depth_matrix_next), after {which}')
#         plt.xlabel('Columns')
#         plt.ylabel('Rows')
#         plt.show()   
# =============================================================================
          
    def pre_loop(self):
        self.populate_from_HDF()
        self.find_num_rows_cols_in_HECRAS()
        self.one_matrix_depth_WSE()
        self.load_tiff_data()
        self.calculate_num_patches()
        self.populate_tiff_patches() # self.pacthes_tiff is populated once and for all
        self.populate_depth_patches() # self.pacthes_depth is populated once and for all 
        self.load_trained_model()
    def divide_forward_reconstruct(self, which):
        self.populate_depth_next_patches(which) 
        """
        takes self.depth_matrix_next and builds the pacthes self.patches_depth_next_dict all types (A, B, C, D). Used var_patches
        """
        self.forward_pass(which) # self.patches_depth_next_dict[which] 
        """
        populates self.populate_true_depth_next_patches(which)
        outputs = self.patches_true_depth_next_dict[which] - self.patches_depth_dict[which]
        self.patches_depth_next_dict[which] = outputs_zeroed + self.patches_depth_dict[which]
        
        """
        self.reconstruct_depth_matrix_next(which) # updates self.depth_matrix_next
    def divide_forward_reconstruct_ABCD(self):
        for which in ['A', 'B', 'C', 'D']:
            self.divide_forward_reconstruct(which)
            self.plot_diff_prediction_and_true()
      
    def closure_loop(self):
        self.pre_loop()

        # Initialize lists to collect metrics
        next_old_diffs = []
        true_pred_diffs = []
    
    
        dummy_performance = np.linalg.norm(self.depth_matrix_next_dummy - self.saved_true_matrix_next)
        print('dummy_performance: ', dummy_performance)
        true_pred_diff = np.linalg.norm(self.saved_true_matrix_next - self.depth_matrix_next)
        print('true_pred_diff: ', true_pred_diff)
        true_pred_diffs.append(true_pred_diff) 

        depth_matrix_next_old = copy.deepcopy(self.depth_matrix_next)
        self.divide_forward_reconstruct_ABCD()
    
        next_old_diff = np.linalg.norm(depth_matrix_next_old - self.depth_matrix_next)
        next_old_diffs.append(next_old_diff)
        print('next_old_diff: ', next_old_diff)

        true_pred_diff = np.linalg.norm(self.saved_true_matrix_next - self.depth_matrix_next)
        true_pred_diffs.append(true_pred_diff) 
        print('true_pred_diff: ', true_pred_diff)

        while (true_pred_diffs[-1] < true_pred_diffs[-2]):
        #while (np.linalg.norm(depth_matrix_next_old - self.depth_matrix_next) > self.tolerance):
            depth_matrix_next_old = copy.deepcopy(self.depth_matrix_next)
            self.divide_forward_reconstruct_ABCD()
            
            depth_matrix_next_old = copy.deepcopy(self.depth_matrix_next)
            self.divide_forward_reconstruct_ABCD()
        
            next_old_diff = np.linalg.norm(depth_matrix_next_old - self.depth_matrix_next)
            next_old_diffs.append(next_old_diff)
            print('next_old_diffs: ', next_old_diff)
    
            true_pred_diff = np.linalg.norm(self.saved_true_matrix_next - self.depth_matrix_next)
            true_pred_diffs.append(true_pred_diff) 
            print('true_pred_diffs: ', true_pred_diff)
            
        # Plot the collected metrics
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        
        ax1.plot(next_old_diffs, label='Next - Old Difference')
        ax1.set_title('Next - Old Difference')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Difference')
        ax1.legend()
        
        ax2.plot(true_pred_diffs, label='True - Prediction Difference')
        ax2.axhline(y=dummy_performance, color='r', linestyle='--', label='Dummy Performance')
        ax2.set_title('True - Prediction Difference')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Difference')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


    def test_convergence(self):
        print('test convergence = ', np.linalg.norm(self.saved_true_matrix_next - self.depth_matrix_next, ord=1))
    def check_overnight(self):
        self.pre_loop()
        self.divide_forward_reconstruct_ABCD()
        while (np.linalg.norm(self.saved_true_matrix_next - self.depth_matrix_next)>self.tolerance):
            print('np.linalg.norm', np.linalg.norm(self.saved_true_matrix_next - self.depth_matrix_next, ord=1))
            self.divide_forward_reconstruct_ABCD()

    def plot_prediction_and_true(self):
        
        import matplotlib.pyplot as plt

        # Calculate the difference between the two matrices
        difference_matrix = self.depth_matrix_next - self.saved_true_matrix_next
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        
        # Plot depth_matrix_next
        im1 = axes[0].imshow(self.depth_matrix_next, cmap='Blues')
        axes[0].set_title('Depth Matrix Next')
        fig.colorbar(im1, ax=axes[0], orientation='vertical')

        # Plot saved_true_matrix_next
        im2 = axes[1].imshow(self.saved_true_matrix_next, cmap='Blues')
        axes[1].set_title('Saved True Matrix Next')
        fig.colorbar(im2, ax=axes[1], orientation='vertical')

        # Plot difference_matrix
        im3 = axes[2].imshow(difference_matrix, cmap='Blues')
        axes[2].set_title('Difference Matrix')
        fig.colorbar(im3, ax=axes[2], orientation='vertical')
        
        plt.tight_layout()
        plt.show()
    def plot_diff_prediction_and_true(self):
        import matplotlib.pyplot as plt
    
        # Calculate the difference between the two matrices
        difference_matrix = self.depth_matrix_next - self.saved_true_matrix_next
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot difference_matrix
        im = ax.imshow(difference_matrix, cmap='Blues')
        ax.set_title('Difference Matrix = depth_matrix_next - saved_true_matrix_next')
        fig.colorbar(im, ax=ax, orientation='vertical')
        
        plt.tight_layout()
        plt.show()
    def plot_prediction_and_dummy(self):
            
            import matplotlib.pyplot as plt

            # Calculate the difference between the two matrices
            difference_matrix = self.depth_matrix_next - self.depth_matrix_next_dummy
            
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
            
            # Plot depth_matrix_next
            im1 = axes[0].imshow(self.depth_matrix_next, cmap='Blues')
            axes[0].set_title('Depth Matrix Next')
            fig.colorbar(im1, ax=axes[0], orientation='vertical')

            # Plot saved_true_matrix_next
            im2 = axes[1].imshow(self.depth_matrix_next_dummy, cmap='Blues')
            axes[1].set_title('Dummy Next')
            fig.colorbar(im2, ax=axes[1], orientation='vertical')

            # Plot difference_matrix
            im3 = axes[2].imshow(difference_matrix, cmap='Blues')
            axes[2].set_title('Difference Matrix')
            fig.colorbar(im3, ax=axes[2], orientation='vertical')
            
            plt.tight_layout()
            plt.show()
    def plot_dummy_and_true(self):
        

        # Calculate the difference between the two matrices
        difference_matrix = self.depth_matrix_next_dummy - self.saved_true_matrix_next
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        
        # Plot depth_matrix_next
        im1 = axes[0].imshow(self.depth_matrix_next_dummy, cmap='Blues')
        axes[0].set_title('Depth Matrix Next Dummy')
        fig.colorbar(im1, ax=axes[0], orientation='vertical')

        # Plot saved_true_matrix_next
        im2 = axes[1].imshow(self.saved_true_matrix_next, cmap='Blues')
        axes[1].set_title('Saved True Matrix Next')
        fig.colorbar(im2, ax=axes[1], orientation='vertical')

        # Plot difference_matrix
        im3 = axes[2].imshow(difference_matrix, cmap='Blues')
        axes[2].set_title('Difference Matrix')
        fig.colorbar(im3, ax=axes[2], orientation='vertical')
        
        plt.tight_layout()
        plt.show()
    def plot_all_matrices(self):
        # Calculate the difference between the two sets of matrices
        difference_matrix = self.depth_matrix_next - self.saved_true_matrix_next
        print('sum_of_diff_matrix_first_row',np.sum(np.abs(difference_matrix)))
        difference_matrix_dummy = self.depth_matrix_next_dummy - self.saved_true_matrix_next
        print('sum_of_diff_matrix_second_row',np.sum(np.abs(difference_matrix_dummy)))


        # Find the y-limits for the depth maps and difference maps
        depth_vmin = min(self.depth_matrix_next.min(), self.saved_true_matrix_next.min(), 
                         self.depth_matrix_next_dummy.min(), self.saved_true_matrix_next.min())
        depth_vmax = max(self.depth_matrix_next.max(), self.saved_true_matrix_next.max(), 
                         self.depth_matrix_next_dummy.max(), self.saved_true_matrix_next.max())
        
        diff_vmin = min(difference_matrix.min(), difference_matrix_dummy.min())
        diff_vmax = max(difference_matrix.max(), difference_matrix_dummy.max())

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
        
        # Plot depth_matrix_next
        im1 = axes[0, 0].imshow(self.depth_matrix_next, cmap='Blues', vmin=depth_vmin, vmax=depth_vmax)
        axes[0, 0].set_title('Depth Matrix Next')
        fig.colorbar(im1, ax=axes[0, 0], orientation='vertical')

        # Plot saved_true_matrix_next
        im2 = axes[0, 1].imshow(self.saved_true_matrix_next, cmap='Blues', vmin=depth_vmin, vmax=depth_vmax)
        axes[0, 1].set_title('Saved True Matrix Next')
        fig.colorbar(im2, ax=axes[0, 1], orientation='vertical')

        # Plot difference_matrix
        im3 = axes[0, 2].imshow(difference_matrix, cmap='Blues', vmin=diff_vmin, vmax=diff_vmax)
        axes[0, 2].set_title('Difference Matrix')
        fig.colorbar(im3, ax=axes[0, 2], orientation='vertical')

        # Plot depth_matrix_next_other
        im4 = axes[1, 0].imshow(self.depth_matrix_next_dummy, cmap='Blues', vmin=depth_vmin, vmax=depth_vmax)
        axes[1, 0].set_title('Depth Matrix Next Dummy')
        fig.colorbar(im4, ax=axes[1, 0], orientation='vertical')

        # Plot saved_true_matrix_next_other
        im5 = axes[1, 1].imshow(self.saved_true_matrix_next, cmap='Blues', vmin=depth_vmin, vmax=depth_vmax)
        axes[1, 1].set_title('Saved True Matrix Next')
        fig.colorbar(im5, ax=axes[1, 1], orientation='vertical')

        # Plot difference_matrix_other
        im6 = axes[1, 2].imshow(difference_matrix_dummy, cmap='Blues', vmin=diff_vmin, vmax=diff_vmax)
        axes[1, 2].set_title('Difference Matrix Dummy')
        fig.colorbar(im6, ax=axes[1, 2], orientation='vertical')
        
        plt.tight_layout()
        plt.show()
    def calculate_mae(self, prediction, truth):
        return np.mean(np.abs(prediction - truth))
    def calculate_mse(self, prediction, truth):
        return np.mean((prediction - truth) ** 2)
    def print_error_metrics(self):
        mae_pred = self.calculate_mae(self.depth_matrix_next, self.saved_true_matrix_next)
        mse_pred = self.calculate_mse(self.depth_matrix_next, self.saved_true_matrix_next)
        mae_dummy = self.calculate_mae(self.depth_matrix_next_dummy, self.saved_true_matrix_next)
        mse_dummy = self.calculate_mse(self.depth_matrix_next_dummy, self.saved_true_matrix_next)
        
        print(f"MAE (Prediction): {mae_pred}")
        print(f"MSE (Prediction): {mse_pred}")
        print(f"MAE (Dummy): {mae_dummy}")
        print(f"MSE (Dummy): {mse_dummy}")

#%% Iterate until convergence
my_closure = Deep_Closure()
my_closure.closure_loop()
my_closure.test_convergence()
#print('#################################### CHECK OVERNIGHT')
#my_closure.check_overnight()
#%% plots
my_closure.plot_prediction_and_true()
my_closure.plot_diff_prediction_and_true()
my_closure.plot_dummy_and_true()
my_closure.plot_prediction_and_dummy()
my_closure.plot_all_matrices()
my_closure.print_error_metrics()
#%% 10 simulations from prj_02
prj_num ='02'
prj_name='HECRAS_on_02' # must be check it for each project
plan_nums = ['02', '03','04','05', '08', '09', '10', '11', '13','14']
#%% closure on prj_02
for plan_num in plan_nums:
    my_closure = Deep_Closure(prj_name= prj_name, prj_num=prj_num, plan_num=plan_num)
    my_closure.check_overnight()
    my_closure.test_convergence()
    my_closure.print_error_metrics()
    my_closure.plot_prediction_and_true()
    my_closure.plot_diff_prediction_and_true()
    my_closure.plot_dummy_and_true()
    my_closure.plot_prediction_and_dummy()
    my_closure.plot_all_matrices()
#%% 10 simulations from prj_03 
prj_num ='03'
prj_name='hecras_on_03' # must be check it for each project
plan_nums= ['01', '02', '03', '04', '05', '06', '07', '08', '09']#, '10']
#%% closure on prj_03
for plan_num in plan_nums:
    my_closure = Deep_Closure(prj_name= prj_name, prj_num=prj_num, plan_num=plan_num)
    my_closure.closure_loop()
    my_closure.test_convergence()
    my_closure.print_error_metrics()
#%% 
def test_var_patches(trimmmed_var_data, num_patches_row, num_patches_col, num_patches, cells_in_patch):
    patches_var = np.zeros((num_patches, cells_in_patch, cells_in_patch))
    # just like from from_HDF
    row_indices = np.arange(0, num_patches_row, 1) * cells_in_patch # on a paper think, suppose each patch holds 10 cells, then the indices would be [0-9, 10-19, 20,29..]. so the indices are [0, 10, 20]. 10 is cells_in_patch
    col_indices =  np.arange(0, num_patches_col, 1) * cells_in_patch
    rows, cols = np.meshgrid(row_indices, col_indices)
    indices = np.column_stack((rows.ravel(), cols.ravel()))

    for i, (row, col) in enumerate(indices):
        patch = trimmmed_var_data[row:row+cells_in_patch, col:col+cells_in_patch] 
        patches_var[i] =  copy.deepcopy(patch)
        
    return patches_var


# Create a matrix with integers from 1, 2, 3,... in a row-major format
num_patches_row = 4
num_patches_col = 3
cells_in_patch = 4

total_rows = num_patches_row * cells_in_patch
total_cols = num_patches_col * cells_in_patch

matrix = np.arange(1, total_rows * total_cols + 1).reshape((total_rows, total_cols))

# Apply the test_var_patches function
num_patches = num_patches_row * num_patches_col
patches = test_var_patches(matrix, num_patches_row, num_patches_col, num_patches, cells_in_patch)

# Display the patches
for i, patch in enumerate(patches):
    print(f"Patch {i+1}:\n{patch}\n")
#%%
def test_reconstruct_depth_matrix_next(patches, num_patches_row, num_patches_col, cells_in_patch):
    # Create an empty array for the reconstructed depth matrix
    depth_next_matrix = np.zeros((num_patches_row * cells_in_patch, num_patches_col * cells_in_patch))

    for i in range(num_patches_col):
        for j in range(num_patches_row):
            patch_index = i * num_patches_row + j
            row_start = j * cells_in_patch
            row_end = (j + 1) * cells_in_patch
            col_start = i * cells_in_patch
            col_end = (i + 1) * cells_in_patch
            depth_next_matrix[row_start:row_end, col_start:col_end] = patches[patch_index]
    
    return depth_next_matrix

# Create a matrix with integers from 1, 2, 3,... in a row-major format
num_patches_row = 3
num_patches_col = 4
cells_in_patch = 10

total_rows = num_patches_row * cells_in_patch
total_cols = num_patches_col * cells_in_patch

matrix = np.arange(1, total_rows * total_cols + 1).reshape((total_rows, total_cols))

# Apply the test_var_patches function
num_patches = num_patches_row * num_patches_col
patches = test_var_patches(matrix, num_patches_row, num_patches_col, num_patches, cells_in_patch)

# Reconstruct the matrix from the patches
reconstructed_matrix = test_reconstruct_depth_matrix_next(patches, num_patches_row, num_patches_col, cells_in_patch)

# Display the original and reconstructed matrices
print("Original Matrix:\n", matrix)
print("\nReconstructed Matrix:\n", reconstructed_matrix)

# Check if the original and reconstructed matrices are the same
if np.array_equal(matrix, reconstructed_matrix):
    print("\nReconstruction successful, the matrices are equal!")
else:
    print("\nReconstruction failed, the matrices are not equal.")