#%% Import Libraries
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

# Set the font to Nimbus Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman']

#%% settings constant
# Set the default tensor type to float64
torch.set_default_dtype(torch.float64) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Import Functions and Classes 
sys.path.append('/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Functions')
from TerrainDownsample_k11s1p0 import TerrainDownsample_k11s1p0
from arch_05 import arch_05

#%% Deep_Closure class
class Deep_Closure:
    def __init__(self,
                 prj_num='03',
                 prj_name='hecras_on_03', # must be check it for each project
                 plan_num='14',
                 cells_in_patch=32,
                 t = 1002,
                 tolerance = 10e-5,
                 delta_t = 60): # must iterate over plan number. create a list for all plan numbers.

        # HDF file name and location
        self.prj_num = prj_num
        self.prj_name = prj_name
        self.plan_num = plan_num

        self.prj_path = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/HECRAS_Simulations_Results/prj_' + self.prj_num 
        self.terrain_path = self.prj_path + '/Terrains'

        self.plan_file_name = prj_name + '.p' + self.plan_num + '.hdf'
        self.tiff_name = 'terrain_'+plan_num+'.tif'

        self.meters_in_cell = 10 # even number and must stay even 
        
        self.cells_in_patch = cells_in_patch
        
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
        
        self.patches_tiff_dict = {'A': None,
                                  'B': None,
                                  'C': None,
                                  'D': None} 
        
        self.saved_initial_BD = None
        self.trimmed_1km_inwards = False
   
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
        
        for t in range(len(self.depth_vectors) - (self.delta_t)):
            # Calculate the difference between consecutive time steps
            diff = self.depth_vectors[t] - self.depth_vectors[t + self.delta_t]
            
            # Check if the maximum difference is less than 1
            if np.max(np.abs(diff)) < 1:
                # Store the index
                valid_time_indices.append(t)
            
            
    def find_num_rows_cols_in_HECRAS(self): 
        threshold = 1  # Define a threshold for significant difference
        # Count the number of rows where the second entry is close to cells_center_coords[0][1]
        count = sum(1 for row in self.cells_center_coords if abs(row[1] - self.cells_center_coords[0][1]) < threshold)
        self.num_cols = count - 2 # HECRAS used outer cells 
        # Count the number of rows where the second entry is close to cells_center_coords[0][0]
        count = sum(1 for row in self.cells_center_coords if abs(row[0] - self.cells_center_coords[0][0]) < threshold)
        self.num_rows = count - 2 # HECRAS used outer cells 

        self.num_cells = self.num_rows * self.num_cols
        
    def one_matrix_depth(self): 
        depth_vector = self.depth_vectors[self.t]
        depth_vector_next = self.depth_vectors[self.t + self.delta_t]
        
        depth_vector = depth_vector[ :self.num_cells]
        depth_vector_next = depth_vector_next[ :self.num_cells]   
        
        new_shape = (self.num_rows, self.num_cols) 
        #np.reshape works exactly like HECRAS stores cells (row major manner)
        self.depth_matrix = np.reshape(depth_vector, new_shape) # (num_rows, num_cols)
        self.depth_matrix_next = np.reshape(depth_vector_next, new_shape) # (num_rows, num_cols)
        
        # trim  cells 1 km inwards 
        if self.num_rows > 240 and self.num_cols > 240:
            self.trimmed_1km_inwards = True
            print('Trimming Water Depths')
            self.depth_matrix = self.depth_matrix[100:-100, 100:-100]
            self.depth_matrix_next = self.depth_matrix_next[100:-100, 100:-100]
            print('num_rows_before_trim', self.num_rows)
            print('num_cols_before_trim', self.num_cols)

            self.num_rows = self.num_rows - 200
            print('num_rows_after_trim', self.num_rows)

            self.num_cols = self.num_cols - 200
            print('num_cols_after_trim', self.num_cols)

            self.num_cells = self.num_rows * self.num_cols
            
    def load_tiff_data(self): 
        os.chdir(self.terrain_path)
        self.tiff_data = tifffile.imread(self.tiff_name).astype(np.float64)
        os.chdir(self.terrain_path)
        self.tiff_data = tifffile.imread(self.tiff_name)
        if self.trimmed_1km_inwards:
            self.tiff_data = self.tiff_data[1000:,1000:] #trimming

    def calculate_num_patches_and_trim(self): 
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
        
        self.depth_matrix = self.depth_matrix[:self.num_rows, :self.num_cols]
        self.depth_matrix_next = self.depth_matrix_next[:self.num_rows, :self.num_cols]
        
        self.saved_true_matrix_next = copy.deepcopy(self.depth_matrix_next)
        self.depth_matrix_next = self.previous_depth_internal(self.depth_matrix_next) # for creating input 
        self.depth_matrix_next_dummy = copy.deepcopy(self.depth_matrix_next)
        
        # when looking at dual patches we throw the first cells_in_patch/2 in both dims, resulting in a decrease in one patch in each dimension.  from there the same computation should be performed. as the last cell must be discarded in both dims to avoid non square cells.
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
        row_indices = np.arange(0, num_patches_row, 1) * (self.tiff_points_in_patch - 1) # on a paper, think, suppose each patch has 6 tiff points, then the indices would be [0-5, 5-10, 10-15, 15-40]. so, the indices are [0,5,15,..]. 5 is tiff_points_in_patch - 1
        col_indices =  np.arange(0, num_patches_col , 1) * (self.tiff_points_in_patch - 1) # just like columns
        rows, cols = np.meshgrid(row_indices, col_indices)
        indices = np.column_stack((rows.ravel(), cols.ravel())) # it provides the indices of patches in a column major manner: [0,0], [10,0], [40,0]
        
        trimmed = copy.deepcopy(trimmed_tiff_data)
        for i, (row, col) in enumerate(indices):
          patch = trimmed[row : row + self.tiff_points_in_patch, col : col + self.tiff_points_in_patch] # row:row+tiff_points_in_patch 0:11 is 0,1,2,3,4,5,7,8,9,10 (exactly as needed)
          patches_tiff[i] = patch - np.min(patch)
         
        return patches_tiff
    
    def populate_tiff_patches(self): 
        #this will be run once and untouched since tiff patches do not change 
        which = 'A'
        self.patches_tiff_dict[which] =  self.tiff_patches(trimmed_tiff_data = self.tiff_data, # no trimming for type A. 
                                                        num_patches_row = self.num_patches_row_dict[which],
                                                        num_patches_col = self.num_patches_col_dict[which],
                                                        num_patches = self.num_patches_dict[which])
    
               
        strts_from = int(self.meters_in_patch/2) #throw tiff points corresponding to the first 5 cells only. which are 0-49. the tiff point at the location of 50, is for the sixth cell.  

        which = 'B'
        self.patches_tiff_dict[which] = self.tiff_patches(trimmed_tiff_data = self.tiff_data[strts_from:-strts_from, strts_from:-strts_from],
                                                     num_patches_row = self.num_patches_row_dict[which],
                                                     num_patches_col = self.num_patches_col_dict[which],
                                                     num_patches = self.num_patches_dict[which])
        
        
        which = 'C'
        self.patches_tiff_dict[which] = self.tiff_patches(trimmed_tiff_data = self.tiff_data[:, strts_from:-strts_from], #must doubles check the trimming
                                                     num_patches_row = self.num_patches_row_dict[which],
                                                     num_patches_col = self.num_patches_col_dict[which],
                                                     num_patches = self.num_patches_dict[which])
        
        which = 'D'
        self.patches_tiff_dict[which] = self.tiff_patches(trimmed_tiff_data = self.tiff_data[strts_from:-strts_from, :], #must doubles check the trimming
                                                    num_patches_row = self.num_patches_row_dict[which],
                                                    num_patches_col = self.num_patches_col_dict[which],
                                                    num_patches = self.num_patches_dict[which])  
        
        
    def depth_patches(self, trimmmed_var_data, num_patches_row, num_patches_col, num_patches):
        patches_var = np.zeros((num_patches, self.cells_in_patch, self.cells_in_patch), dtype=np.float64)
        row_indices = np.arange(0, num_patches_row, 1) * self.cells_in_patch # on a paper think, suppose each patch holds 10 cells, then the indices would be [0-9, 10-19, 40,29..]. so the indices are [0, 10, 40]. 10 is cells_in_patch
        col_indices =  np.arange(0, num_patches_col, 1) * self.cells_in_patch
        rows, cols = np.meshgrid(row_indices, col_indices)
        indices = np.column_stack((rows.ravel(), cols.ravel()))

        for i, (row, col) in enumerate(indices):
            patch = trimmmed_var_data[row:row+self.cells_in_patch, col:col+self.cells_in_patch] 
            patches_var[i] =  copy.deepcopy(patch)
           
        return patches_var
    
    def populate_depth_patches(self): 
        # this function will be run only one, since depth matrix is not update at n+1
        depth_matrix = copy.deepcopy(self.depth_matrix)
        which = 'A'
        self.patches_depth_dict[which] =  self.depth_patches(
                                                              trimmmed_var_data = depth_matrix,
                                                              num_patches_row = self.num_patches_row_dict[which],
                                                              num_patches_col = self.num_patches_col_dict[which],
                                                              num_patches = self.num_patches_dict[which])
            
        strts_from = int(self.cells_in_patch/2)
        
        which = 'B'
        self.patches_depth_dict[which] = self.depth_patches(
                                                            trimmmed_var_data = depth_matrix[strts_from:-strts_from, strts_from:-strts_from], #must doubles check the trimming
                                                            num_patches_row = self.num_patches_row_dict[which],
                                                            num_patches_col = self.num_patches_col_dict[which],
                                                            num_patches = self.num_patches_dict[which])

        which = 'C'
        self.patches_depth_dict[which]= self.depth_patches(
                                                            trimmmed_var_data = depth_matrix[:, strts_from:-strts_from], #must doubles check the trimming
                                                            num_patches_row = self.num_patches_row_dict[which],
                                                            num_patches_col = self.num_patches_col_dict[which],
                                                            num_patches = self.num_patches_dict[which])

        which = 'D'
        self.patches_depth_dict[which] = self.depth_patches(
                                                            trimmmed_var_data = depth_matrix[strts_from:-strts_from, :], #must doubles check the trimming
                                                            num_patches_row = self.num_patches_row_dict[which],
                                                            num_patches_col = self.num_patches_col_dict[which],
                                                            num_patches = self.num_patches_dict[which])
        
    def populate_depth_next_patches(self, which): 
        depth_matrix_next = copy.deepcopy(self.depth_matrix_next)    
        if which == 'A':
            self.patches_depth_next_dict[which] =  self.depth_patches(
                                                                    trimmmed_var_data = depth_matrix_next,
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])

        strts_from = int(self.cells_in_patch/2)
        
        if which == 'B':
            self.patches_depth_next_dict[which] = self.depth_patches(
                                                                    trimmmed_var_data = depth_matrix_next[strts_from:-strts_from, strts_from:-strts_from], # removes first and last strts_from rows and cols
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])
            
        if which == 'C':
            self.patches_depth_next_dict[which] = self.depth_patches(
                                                                    trimmmed_var_data = depth_matrix_next[:, strts_from:-strts_from], #must doubles check the trimming
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])
        if which == 'D':
            self.patches_depth_next_dict[which] = self.depth_patches(
                                                                    trimmmed_var_data = depth_matrix_next[strts_from:-strts_from, :], #must doubles check the trimming
                                                                    num_patches_row = self.num_patches_row_dict[which],
                                                                    num_patches_col = self.num_patches_col_dict[which],
                                                                    num_patches = self.num_patches_dict[which])
    # HELPING FUNCTIONS                                   
    def zero_internal(self, matrix):
        matrix_to_change =  copy.deepcopy(matrix)
        rows, cols = matrix_to_change.shape
        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                matrix_to_change[i][j] = 0
        return matrix_to_change
    
    def previous_depth_internal(self, matrix):
        matrix_to_change = copy.deepcopy(matrix)
        depth_matrix = copy.deepcopy(self.depth_matrix)
        rows, cols = matrix_to_change.shape
        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                matrix_to_change[i][j] = depth_matrix[i][j]
        return matrix_to_change
     
    def depth_tensors_model_input(self, which):
        patches_depth = copy.deepcopy(self.patches_depth_dict[which])
        patches_depth_next = copy.deepcopy(self.patches_depth_next_dict[which]) # self.patches_depth_next_dict[which] is (num_patches[which], cells_in_patch, cells_in_patch)
        patches_depth_next_BC = [self.zero_internal(matrix) for matrix in patches_depth_next]
        
        patches_data = []
        
        num_patches = self.num_patches_dict[which]
        
        for i in range(num_patches):
            concatenated_patch = np.stack((patches_depth_next_BC[i], patches_depth[i]), axis=0) # stacking as opposed to concatenation adds a new dim
            patches_data.append(concatenated_patch)
    
        patches_data = np.array(patches_data, dtype=np.float64) # convert list into an array
        return torch.tensor(patches_data, dtype=torch.double).cuda() # torch tensor (N, 2, cells_in_patch, cells_in_patch)
    
    def tiff_tensors_model_input(self, which):
        tiff_patches = copy.deepcopy(self.patches_tiff_dict[which])        
        tiff_patches = np.expand_dims(tiff_patches, axis=1)
        return torch.tensor(tiff_patches, dtype=torch.double).cuda() # (N, 1, tiff_point_in_patch, tiff_point_in_patch)
    
    def load_trained_model(self):
        # Initialize models
        trial_num = 'trial_final'
        Architecture_num = 'Arch_05'
        down_c_start = 1
        down_c1 = 10
        down_c2 = 20
        down_c_end = 1
        
        shared_terrain_downsample = TerrainDownsample_k11s1p0(down_c_start, down_c_end, down_c1, down_c2).to(device)
        self.model = arch_05(shared_terrain_downsample).to(device)
        print("Trained model loaded successfully.")
        base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final'
        model_dir = os.path.join(base_dir, f'{Architecture_num}/{trial_num}/trained_models')
        model_path = os.path.join(model_dir, 'model.pth')
        # Load the model's state dictionary
        folder = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/{Architecture_num}/{trial_num}/trained_models/'
        model_name = 'model'
        self.model.load_state_dict(torch.load(model_path))

        # Set the model to evaluation mode (if you plan to use it for inference)
        print("Model loaded and ready for inference") 
        
    def forward_pass(self, which):  # MUST WORK ON IT TONIGHT
        # Create tensors from depth_an_BC and terrain
        depth_input = self.depth_tensors_model_input(which)
        terrain_input = self.tiff_tensors_model_input(which)
       
        self.model.eval()
        
        outputs = self.model(terrain_input, depth_input) # output is a tensor (N, 1, num_cells, num_cells)
        outputs_squeezed = outputs.squeeze(dim=1)  # remove the dimension at dim=1
        outputs_numpy = outputs_squeezed.cpu().detach().numpy()  # convert the tensor to a NumPy array
        
        self.patches_depth_next_dict[which] = outputs_numpy + self.patches_depth_dict[which]    
            
    def reconstruct_depth_matrix_next(self, which): 
        patches = self.patches_depth_next_dict[which] # this assumes patches_depth_next_dict is udpated after forward pass
        num_patches_row = self.num_patches_row_dict[which]
        num_patches_col = self.num_patches_col_dict[which]

        # Create an empty array for the reconstructed depth matrix
        depth_next_matrix = np.zeros((num_patches_row * self.cells_in_patch, num_patches_col * self.cells_in_patch), dtype=np.float64)

        # populate this empty matrix with the output
        for i in range(num_patches_col): # num patches in col dimension (in one row)
            for j in range(num_patches_row): # num patches in row dimension (in one col)
                patch_index = i * num_patches_row + j
                row_start = j * self.cells_in_patch
                row_end = (j + 1) * self.cells_in_patch
                col_start = i * self.cells_in_patch
                col_end = (i + 1) * self.cells_in_patch
                depth_next_matrix[row_start:row_end, col_start:col_end] = patches[patch_index]
        
        strts_from = int(self.cells_in_patch/2)
        
        if which == 'A':
            self.depth_matrix_next[2:-2, 2:-2] = depth_next_matrix[2:-2, 2:-2] # you want to avoid touching real BC
            
        if which == 'B':
            self.depth_matrix_next[strts_from:-strts_from, strts_from:-strts_from] = depth_next_matrix
            
        if which == 'C':
            self.depth_matrix_next[2:-2, strts_from:-strts_from] =  depth_next_matrix[2:-2, :]
            
        if which == 'D':
            self.depth_matrix_next[strts_from:-strts_from, 2:-2] = depth_next_matrix[:, 2:-2]
          
    def pre_loop(self):
        self.populate_from_HDF()
        self.find_num_rows_cols_in_HECRAS()
        self.one_matrix_depth()
        self.load_tiff_data()
        self.calculate_num_patches_and_trim()
        self.populate_tiff_patches() # self.pacthes_tiff is populated once and for all
        self.populate_depth_patches() # self.pacthes_depth is populated once and for all 
        self.load_trained_model()
        
    def divide_forward_reconstruct(self, which):
        self.populate_depth_next_patches(which) 
        self.forward_pass(which) # self.patches_depth_next_dict[which] 
        self.reconstruct_depth_matrix_next(which) # updates self.depth_matrix_next
        
    def divide_forward_reconstruct_ABCD(self):
        for which in ['A', 'B', 'C', 'D']:
            self.divide_forward_reconstruct(which)
      
    def closure_loop(self):
        self.pre_loop()
    
        # Initialize lists to collect metrics
        next_old_diffs = []
        true_pred_diffs = []
    
        # Calculate the dummy performance using MAE
        dummy_performance = mean_absolute_error(self.saved_true_matrix_next.flatten(), self.depth_matrix_next_dummy.flatten())
    
        # Calculate the initial true prediction difference using MAE
        true_pred_diff = mean_absolute_error(self.saved_true_matrix_next.flatten(), self.depth_matrix_next.flatten())
        true_pred_diffs.append(true_pred_diff) 
    
        depth_matrix_next_old = copy.deepcopy(self.depth_matrix_next)
        self.divide_forward_reconstruct_ABCD()
    
        # Calculate the next-old difference using MAE
        next_old_diff = mean_absolute_error(depth_matrix_next_old.flatten(), self.depth_matrix_next.flatten())
        next_old_diffs.append(next_old_diff)
    
        # Calculate the true-prediction difference using MAE
        true_pred_diff = mean_absolute_error(self.saved_true_matrix_next.flatten(), self.depth_matrix_next.flatten())
        true_pred_diffs.append(true_pred_diff) 
        
        while (mean_absolute_error(depth_matrix_next_old.flatten(), self.depth_matrix_next.flatten()) > self.tolerance and len(next_old_diffs)<30):
            
            depth_matrix_next_old = copy.deepcopy(self.depth_matrix_next)
            self.divide_forward_reconstruct_ABCD()
    
            next_old_diff = mean_absolute_error(depth_matrix_next_old.flatten(), self.depth_matrix_next.flatten())
            next_old_diffs.append(next_old_diff)
    
            true_pred_diff = mean_absolute_error(self.saved_true_matrix_next.flatten(), self.depth_matrix_next.flatten())
            true_pred_diffs.append(true_pred_diff) 
        
        L1_test.append(true_pred_diffs[-1]) 
        RAE = true_pred_diff/dummy_performance
        RAE_test.append(RAE)
        
        # Define the path to save L1 values
        save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/L1_test/{prj_num}_{self.plan_num}_{self.t}_L1.npy'
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the array of L1 values to a file
        np.save(save_path, L1_test[-1])

        # Define the path to save L1 values
        save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/RAE_test/{prj_num}_{self.plan_num}_{self.t}_RAE.npy'
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the array of L1 values to a file
        np.save(save_path, RAE_test[-1])
        
        # Plot the collected metrics
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(30, 10), gridspec_kw={'wspace': 0.3})
        
        fontsize = 30  # Set font size same as plot_all_matrices
        ax1.plot(next_old_diffs, label='Predicted_Current - Predicted_Old')
        ax1.set_title('|Predicted_Current - Predicted_Old|', fontsize=fontsize)
        ax1.set_xlabel('Iteration', fontsize=fontsize)
        ax1.set_ylabel('Mean Absolute Difference (m)', fontsize=fontsize)
        ax1.legend(fontsize=fontsize)
        ax1.tick_params(labelsize=fontsize)
        
        ax2.plot(true_pred_diffs, label='|Ground_Truth - Predicted_Current|')
        ax2.axhline(y=dummy_performance, color='r', linestyle='--', label='|Ground_Truth - Dummy|')
        ax2.set_title('|Ground_Truth - Predicted_Current|', fontsize=fontsize)
        ax2.set_xlabel('Iteration', fontsize=fontsize)
        ax2.set_ylabel('Mean Absolute Error (m)', fontsize=fontsize)
        ax2.legend(fontsize=fontsize)
        ax2.tick_params(labelsize=fontsize)
        
        # Define the save path
        save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/Converging/{self.prj_num}_{self.plan_num}_{self.t}.png'
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


    def plot_all_matrices(self):
        # Calculate the difference between the two sets of matrices
        prediction_minus_dummy = self.depth_matrix_next - self.depth_matrix_next_dummy
    
        # Find the y-limits for the depth maps and difference maps
        depth_vmin = min(self.depth_matrix_next.min(), self.saved_true_matrix_next.min(), 
                         self.depth_matrix_next_dummy.min(), self.saved_true_matrix_next.min())
        depth_vmax = max(self.depth_matrix_next.max(), self.saved_true_matrix_next.max(), 
                         self.depth_matrix_next_dummy.max(), self.saved_true_matrix_next.max())
        # Calculate the absolute difference for the custom colormap
        abs_difference_matrix = np.abs(self.saved_true_matrix_next - self.depth_matrix_next)
        abs_difference_matrix_dummy = np.abs(self.saved_true_matrix_next - self.depth_matrix_next_dummy)
        diff_abs_max = max(abs_difference_matrix.max(), abs_difference_matrix_dummy.max())
        
        # Plot setup
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(80, 55))
        fraction = 0.33
        fontsize = 100
        colorbar_ticksize = 60
        white_red_cmap = mcolors.LinearSegmentedColormap.from_list("white_red", [(1, 1, 1), (1, 0, 0)])
        abs_prediction_minus_dummy = np.abs(prediction_minus_dummy)
        greens_cmap = mcolors.LinearSegmentedColormap.from_list("greens", [(0.9, 1, 0.9), (0, 0.5, 0)])
    
        # Define tick interval
        tick_interval = 64
    
        # Plot depth_matrix_next
        im1 = axes[0, 0].imshow(self.depth_matrix_next, cmap='Blues', vmin=depth_vmin, vmax=depth_vmax)
        axes[0, 0].set_title('Predicted Depth Next (m)', fontsize=fontsize)
        cbar1 = fig.colorbar(im1, ax=axes[0, 0], orientation='vertical', fraction=fraction, pad=0.04)
        cbar1.ax.tick_params(labelsize=colorbar_ticksize)
        axes[0, 0].set_xticks(np.arange(0, self.depth_matrix_next.shape[1], tick_interval))
        axes[0, 0].set_yticks(np.arange(0, self.depth_matrix_next.shape[0], tick_interval))
        axes[0, 0].tick_params(axis='both', labelsize=colorbar_ticksize)

    
        # Plot saved_true_matrix_next
        im2 = axes[0, 1].imshow(self.saved_true_matrix_next, cmap='Blues', vmin=depth_vmin, vmax=depth_vmax)
        axes[0, 1].set_title('Ground Truth Next (m)', fontsize=fontsize)
        cbar2 = fig.colorbar(im2, ax=axes[0, 1], orientation='vertical', fraction=fraction, pad=0.04)
        cbar2.ax.tick_params(labelsize=colorbar_ticksize)
        axes[0, 1].set_xticks(np.arange(0, self.saved_true_matrix_next.shape[1], tick_interval))
        axes[0, 1].set_yticks(np.arange(0, self.saved_true_matrix_next.shape[0], tick_interval))
        axes[0, 1].tick_params(axis='both', labelsize=colorbar_ticksize)
    
        # Plot absolute difference_matrix using the white-red colormap
        im3 = axes[0, 2].imshow(abs_difference_matrix, cmap=white_red_cmap, vmin=0, vmax=diff_abs_max)
        axes[0, 2].set_title(f'RAE = {RAE_test[-1]:.2f}\n\n|Truth - Predicted| (m)', fontsize=fontsize)
        cbar3 = fig.colorbar(im3, ax=axes[0, 2], orientation='vertical', fraction=fraction, pad=0.04)
        cbar3.ax.tick_params(labelsize=colorbar_ticksize)
        axes[0, 2].set_xticks(np.arange(0, abs_difference_matrix.shape[1], tick_interval))
        axes[0, 2].set_yticks(np.arange(0, abs_difference_matrix.shape[0], tick_interval))
        axes[0, 2].tick_params(axis='both', labelsize=colorbar_ticksize)
    
        # Plot depth_matrix_next_dummy
        im4 = axes[1, 0].imshow(self.depth_matrix_next_dummy, cmap='Blues', vmin=depth_vmin, vmax=depth_vmax)
        axes[1, 0].set_title('Dummy Depth Next (m)', fontsize=fontsize)
        cbar4 = fig.colorbar(im4, ax=axes[1, 0], orientation='vertical', fraction=fraction, pad=0.04)
        cbar4.ax.tick_params(labelsize=colorbar_ticksize)
        axes[1, 0].set_xticks(np.arange(0, self.depth_matrix_next_dummy.shape[1], tick_interval))
        axes[1, 0].set_yticks(np.arange(0, self.depth_matrix_next_dummy.shape[0], tick_interval))
        axes[1, 0].tick_params(axis='both', labelsize=colorbar_ticksize)
    
        # Plot saved_true_matrix_next
        im5 = axes[1, 1].imshow(self.saved_true_matrix_next, cmap='Blues', vmin=depth_vmin, vmax=depth_vmax)
        axes[1, 1].set_title('Ground Truth Next (m)', fontsize=fontsize)
        cbar5 = fig.colorbar(im5, ax=axes[1, 1], orientation='vertical', fraction=fraction, pad=0.04)
        cbar5.ax.tick_params(labelsize=colorbar_ticksize)
        axes[1, 1].set_xticks(np.arange(0, self.saved_true_matrix_next.shape[1], tick_interval))
        axes[1, 1].set_yticks(np.arange(0, self.saved_true_matrix_next.shape[0], tick_interval))
        axes[1, 1].tick_params(axis='both', labelsize=colorbar_ticksize)

        # Plot absolute difference_matrix_dummy using custom colormap
        im6 = axes[1, 2].imshow(abs_difference_matrix_dummy, cmap=white_red_cmap, vmin=0, vmax=diff_abs_max)
        axes[1, 2].set_title('|Truth - Dummy| (m)', fontsize=fontsize)
        cbar6 = fig.colorbar(im6, ax=axes[1, 2], orientation='vertical', fraction=fraction, pad=0.04)
        cbar6.ax.tick_params(labelsize=colorbar_ticksize)
        axes[1, 2].set_xticks(np.arange(0, abs_difference_matrix_dummy.shape[1], tick_interval))
        axes[1, 2].set_yticks(np.arange(0, abs_difference_matrix_dummy.shape[0], tick_interval))
        axes[1, 2].tick_params(axis='both', labelsize=colorbar_ticksize)

        # Plot depth_matrix_next_dummy again
        im7 = axes[2, 0].imshow(self.depth_matrix_next_dummy, cmap='Blues', vmin=depth_vmin, vmax=depth_vmax)
        axes[2, 0].set_title('Dummy Depth Next (m)', fontsize=fontsize)
        cbar7 = fig.colorbar(im7, ax=axes[2, 0], orientation='vertical', fraction=fraction, pad=0.04)
        cbar7.ax.tick_params(labelsize=colorbar_ticksize)
        axes[2, 0].set_xticks(np.arange(0, self.depth_matrix_next_dummy.shape[1], tick_interval))
        axes[2, 0].set_yticks(np.arange(0, self.depth_matrix_next_dummy.shape[0], tick_interval))
        axes[2, 0].tick_params(axis='both', labelsize=colorbar_ticksize)

    
        # Plot saved_true_matrix_next
        im8 = axes[2, 1].imshow(self.depth_matrix_next, cmap='Blues', vmin=depth_vmin, vmax=depth_vmax)
        axes[2, 1].set_title('Predicted Depth Next (m)', fontsize=fontsize)
        cbar8 = fig.colorbar(im8, ax=axes[2, 1], orientation='vertical', fraction=fraction, pad=0.04)
        cbar8.ax.tick_params(labelsize=colorbar_ticksize)
        axes[2, 1].set_xticks(np.arange(0, self.depth_matrix_next.shape[1], tick_interval))
        axes[2, 1].set_yticks(np.arange(0, self.depth_matrix_next.shape[0], tick_interval))
        axes[2, 1].tick_params(axis='both', labelsize=colorbar_ticksize)
    
        # Plot abs_prediction_minus_dummy with greens colormap
        im9 = axes[2, 2].imshow(abs_prediction_minus_dummy, cmap=greens_cmap, vmin=0, vmax=diff_abs_max)
        axes[2, 2].set_title('|Predicted - Dummy| (m)', fontsize=fontsize)
        cbar9 = fig.colorbar(im9, ax=axes[2, 2], orientation='vertical', fraction=fraction, pad=0.04)
        cbar9.ax.tick_params(labelsize=colorbar_ticksize)
        axes[2, 2].set_xticks(np.arange(0, abs_prediction_minus_dummy.shape[1], tick_interval))
        axes[2, 2].set_yticks(np.arange(0, abs_prediction_minus_dummy.shape[0], tick_interval))
        axes[2, 2].tick_params(axis='both', labelsize=colorbar_ticksize)

        # Save directory for plot_all_matrices
        save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/9_Maps/{self.prj_num}_{self.plan_num}_{self.t}.png'
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

#%% test
all_t = [0, 70, 140, 210]

prj_num= '03'
test_plans = ['61', '63', '65', '71', '72']
prj_name='hecras_on_03'

test_plans = ['38', '39', '40', '41', '42', '44', '45']
prj_num='04'
prj_name='HECRAS'

test_plans = ['84', '85', '86', '87', '88', '89', '90']
prj_num='05'
prj_name='HECRAS'

test_plans = ['44', '45', '46', '47', '48', '49', '50']
prj_num= '06'
prj_name='HECRAS'

#%% prj_03
test_plans = ['61', '63', '65', '71', '72']
prj_num= '03'
prj_name='hecras_on_03'

L1_test = []
RAE_test = []

for curr_plan_num in test_plans:
    for curr_t in all_t:
        try:
            print(f'Current Plan = {curr_plan_num}', f't_start = {curr_t}')
            my_closure = Deep_Closure(
                prj_num=prj_num,
                prj_name=prj_name,  # must be checked for each project
                plan_num=curr_plan_num,
                t=curr_t,
                tolerance=1e-3,
                delta_t=60
            )
            my_closure.closure_loop()
            my_closure.plot_all_matrices()
        except Exception as e:
            print(f'Error encountered with Plan = {curr_plan_num}, t_start = {curr_t}: {e}')
            print('Moving to the next plan...')


    # Define the path to save L1 values
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/L1_test/prj_{prj_num}_L1_test.npy'
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the array of L1 values to a file
    np.save(save_path, np.array(L1_test))
    
    # Define the path to save RAE values
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/RAE_test/prj_{prj_num}_RAE_test.npy'
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the array of L1 values to a file
    np.save(save_path, np.array(RAE_test))

#%% prj_04
test_plans = ['38', '39', '40', '41', '42', '44', '45']
prj_num= '04'
prj_name='HECRAS'

L1_test = []
RAE_test = []

for curr_plan_num in test_plans:
    for curr_t in all_t:
        try:
            print(f'Current Plan = {curr_plan_num}', f't_start = {curr_t}')
            my_closure = Deep_Closure(
                prj_num=prj_num,
                prj_name=prj_name,  # must be checked for each project
                plan_num=curr_plan_num,
                t=curr_t,
                tolerance=1e-3,
                delta_t=60
            )
            my_closure.closure_loop()
            my_closure.plot_all_matrices()
        except Exception as e:
            print(f'Error encountered with Plan = {curr_plan_num}, t_start = {curr_t}: {e}')
            print('Moving to the next plan...')


    # Define the path to save L1 values
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/L1_test/prj_{prj_num}_L1_test.npy'
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the array of L1 values to a file
    np.save(save_path, np.array(L1_test))
    
    # Define the path to save RAE values
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/RAE_test/prj_{prj_num}_RAE_test.npy'
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the array of L1 values to a file
    np.save(save_path, np.array(RAE_test))

#%% prj_05
test_plans = ['84', '85', '86', '87', '88', '89', '90']
prj_num= '05'
prj_name='HECRAS'

L1_test = []
RAE_test = []

for curr_plan_num in test_plans:
    for curr_t in all_t:
        try:
            print(f'Current Plan = {curr_plan_num}', f't_start = {curr_t}')
            my_closure = Deep_Closure(
                prj_num=prj_num,
                prj_name=prj_name,  # must be checked for each project
                plan_num=curr_plan_num,
                t=curr_t,
                tolerance=1e-3,
                delta_t=60
            )
            my_closure.closure_loop()
            my_closure.plot_all_matrices()
        except Exception as e:
            print(f'Error encountered with Plan = {curr_plan_num}, t_start = {curr_t}: {e}')
            print('Moving to the next plan...')


    # Define the path to save L1 values
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/L1_test/prj_{prj_num}_L1_test.npy'
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the array of L1 values to a file
    np.save(save_path, np.array(L1_test))
    
    # Define the path to save RAE values
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/RAE_test/prj_{prj_num}_RAE_test.npy'
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the array of L1 values to a file
    np.save(save_path, np.array(RAE_test))
    
#%% prj_06
test_plans = ['44', '45', '46', '47', '48', '49', '50']
prj_num= '06'
prj_name='HECRAS'

L1_test = []
RAE_test = []

for curr_plan_num in test_plans:
    for curr_t in all_t:
        try:
            print(f'Current Plan = {curr_plan_num}', f't_start = {curr_t}')
            my_closure = Deep_Closure(
                prj_num=prj_num,
                prj_name=prj_name,  # must be checked for each project
                plan_num=curr_plan_num,
                t=curr_t,
                tolerance=1e-3,
                delta_t=60
            )
            my_closure.closure_loop()
            my_closure.plot_all_matrices()
        except Exception as e:
            print(f'Error encountered with Plan = {curr_plan_num}, t_start = {curr_t}: {e}')
            print('Moving to the next plan...')


    # Define the path to save L1 values
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/L1_test/prj_{prj_num}_L1_test.npy'
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the array of L1 values to a file
    np.save(save_path, np.array(L1_test))
    
    # Define the path to save RAE values
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/Closure_Loop/RAE_test/prj_{prj_num}_RAE_test.npy'
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the array of L1 values to a file
    np.save(save_path, np.array(RAE_test))