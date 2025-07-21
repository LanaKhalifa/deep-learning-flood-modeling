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

#%% Deep_Closure class
class Deep_Closure:
    def __init__(self,
                 prj_num='05',
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
        
        self.saved_initial_BD = None
        self.trimmed_1km_inwards = False
        self.delta_t = 60
        self.t = 0
   
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

            self.num_rows = self.num_rows - 200
            print('num_rows_after_trim', self.num_rows)

            self.num_cols = self.num_cols - 200
            print('num_cols_after_trim', self.num_cols)

            self.num_cells = self.num_rows * self.num_cols
            print('num_cells', self.num_cells)


#%% prj_03
prj_num= '03'
test_plans = ['61', '63', '65', '71', '72']
prj_name='hecras_on_03'

for curr_plan_num in test_plans:
    print(f'Current Plan = {curr_plan_num}')
    my_closure = Deep_Closure(
        prj_num=prj_num,
        prj_name=prj_name,  # must be checked for each project
        plan_num=curr_plan_num,
        t=0,
        tolerance=1e-3,
        delta_t=60
    )
    my_closure.populate_from_HDF()
    my_closure.find_num_rows_cols_in_HECRAS()
    my_closure.one_matrix_depth()
    num_cells = my_closure.num_cells
            
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/num_cells/{prj_num}_{curr_plan_num}_num_cells.npy'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.array(num_cells))
#%% prj_num 04
test_plans = ['38', '39', '40', '41', '42', '44', '45']
prj_num='04'
prj_name='HECRAS'

for curr_plan_num in test_plans:
    print(f'Current Plan = {curr_plan_num}')
    my_closure = Deep_Closure(
        prj_num=prj_num,
        prj_name=prj_name,  # must be checked for each project
        plan_num=curr_plan_num,
        t=0,
        tolerance=1e-3,
        delta_t=60
    )
    my_closure.populate_from_HDF()
    my_closure.find_num_rows_cols_in_HECRAS()
    my_closure.one_matrix_depth()
    num_cells = my_closure.num_cells
            
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/num_cells/{prj_num}_{curr_plan_num}_num_cells.npy'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.array(num_cells))

#%% prj_05
test_plans = ['84', '85', '86', '87', '88', '89', '90']
prj_num='05'
prj_name='HECRAS'

for curr_plan_num in test_plans:
    print(f'Current Plan = {curr_plan_num}')
    my_closure = Deep_Closure(
        prj_num=prj_num,
        prj_name=prj_name,  # must be checked for each project
        plan_num=curr_plan_num,
        t=0,
        tolerance=1e-3,
        delta_t=60
    )
    my_closure.populate_from_HDF()
    my_closure.find_num_rows_cols_in_HECRAS()
    my_closure.one_matrix_depth()
    num_cells = my_closure.num_cells
            
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/num_cells/{prj_num}_{curr_plan_num}_num_cells.npy'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.array(num_cells))
#%% prj_06
test_plans = ['44', '45', '46', '47', '48', '49', '50']
prj_num= '06'
prj_name='HECRAS'

for curr_plan_num in test_plans:
    print(f'Current Plan = {curr_plan_num}')
    my_closure = Deep_Closure(
        prj_num=prj_num,
        prj_name=prj_name,  # must be checked for each project
        plan_num=curr_plan_num,
        t=0,
        tolerance=1e-3,
        delta_t=60
    )
    my_closure.populate_from_HDF()
    my_closure.find_num_rows_cols_in_HECRAS()
    my_closure.one_matrix_depth()
    num_cells = my_closure.num_cells
            
    save_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure_new/num_cells/{prj_num}_{curr_plan_num}_num_cells.npy'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, np.array(num_cells))
