#%% Import Libraries
import h5py
import numpy as np
import os
import tifffile
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy import ndimage
import copy
import pickle
import matplotlib.pyplot as plt

#%% from_HDF class
class from_HDF:
    def __init__(self,
                 prj_num='02',
                 prj_name='hecras_on_03', # must be check it for each project
                 plan_num='14',
                 cells_in_patch=32,
                 delta_t = 60): # must iterate over plan number. create a list for all plan numbers.

        # HDF file name and location
        self.prj_num = prj_num
        self.prj_name = prj_name
        self.plan_num = plan_num

        self.prj_path = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/HECRAS_Simulations_Results/prj_' + self.prj_num 
        self.terrain_path = self.prj_path + '/Terrains'

        self.plan_file_name = prj_name + '.p' + self.plan_num + '.hdf'
        self.tiff_name = 'terrain_'+plan_num+'.tif'
        self.k = 4 #num of clusters
        self.delta_t = delta_t
        self.closest_indices = None
        self.meters_in_cell = 10 # even number and must stay even 
        
        if cells_in_patch % 2 == 0:
            self.cells_in_patch = cells_in_patch
        else: 
            raise ValueError("Error: cells_in_patch must be even to allow creating dual patches.")


        self.depth_vectors = None
        self.cells_center_coords = None
        
        self.depth_matrices = None
        self.closest_indices = None
        
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
        
        self.cluster_counter = 0
        
        self.database = {'terrain': [],'depth': [],'depth_next': []}
            
    def h5_tree(self, val, pre=''):
        items = len(val)
        for key, val in val.items():
            items -= 1
            if items == 0:
                # the last item
                if type(val) == h5py._hl.group.Group:
                    print(pre + '└── ' + key)
                    self.h5_tree(val, pre+'    ')
                else:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
            else:
                if type(val) == h5py._hl.group.Group:
                    print(pre + '├── ' + key)
                    self.h5_tree(val, pre+'│   ')
                else:
                    print(pre + '├── ' + key + ' (%d)' % len(val))

    def print_tree(self):
        with h5py.File(self.plan_file_name, 'r') as hf:
            self.h5_tree(hf)

    def from_HDF_file(self):
        with h5py.File(self.plan_file_name,'r') as f:
            from_HDF = {}
            Results = f['Results']['Unsteady']['Output']['Output Blocks']['Base Output']['Unsteady Time Series']['2D Flow Areas']['Perimeter 1']

            from_HDF['Invert_Depth'] = np.array(Results['Cell Invert Depth'])
            from_HDF['Cell Water Surface Error'] = np.array(Results['Cell Water Surface Error'])

            from_HDF['Cells Center Coordinate'] = np.array(f['Geometry']['2D Flow Areas']['Perimeter 1']['Cells Center Coordinate'])
            return from_HDF

    def populate_from_HDF(self):
        os.chdir(self.prj_path)
        from_HDF_dict = self.from_HDF_file()
        self.depth_vectors = from_HDF_dict['Invert_Depth'] # 2D: rows = time dimension. cols = cells
        self.cells_center_coords = from_HDF_dict['Cells Center Coordinate'] # 2D: rows = cells. cols = x, y. 
        self.WSE_error = from_HDF_dict['Cell Water Surface Error']

    def cluster_via_KMeans(self):
        if self.depth_vectors is None:
            print("Error: depth_vectors is not populated. Run populate_from_HDF first.")
            return

        self.closest_indices = [23, 92, 161, 207]
        print('unifrom closest indices: ', self.closest_indices)


    def find_num_rows_cols_in_HECRAS(self):
        threshold = 1  # Define a threshold for significant difference
        # Count the number of rows where the second entry is close to cells_center_coords[0][1]
        count = sum(1 for row in self.cells_center_coords if abs(row[1] - self.cells_center_coords[0][1]) < threshold)
        self.num_cols = count - 2
        # Count the number of rows where the second entry is close to cells_center_coords[0][0]
        count = sum(1 for row in self.cells_center_coords if abs(row[0] - self.cells_center_coords[0][0]) < threshold)
        self.num_rows = count - 2

        self.num_cells = self.num_rows * self.num_cols

    def k_matrices_depth_WSE(self):
        if self.closest_indices is None:
            print("Error: closest_indices is not populated. Run cluster_via_KMeans first.")
            return

        # Initialize k_WSE_vectors and k_depth_vectors with zeros of appropriate shape
        k_depth_vectors = np.zeros((self.k, self.depth_vectors.shape[1]))
        k_depth_vectors_next = np.zeros((self.k, self.depth_vectors.shape[1]))

        # Fill K_WSE_vector with the vectors from WSE_vectors at indices specified by closest_indices
        for i, idx in enumerate(self.closest_indices):
            k_depth_vectors[i] = self.depth_vectors[idx]
            k_depth_vectors_next[i] = self.depth_vectors[idx + self.delta_t]

        k_depth_vectors = k_depth_vectors[:, :self.num_cells]
        k_depth_vectors_next = k_depth_vectors_next[:, :self.num_cells]

        new_shape = (self.k, self.num_rows, self.num_cols)
        
        #np.reshape works exactly like HECRAS stores cells (row major manner)
        self.k_depth_matrices = np.reshape(k_depth_vectors, new_shape)
        self.k_depth_matrices_next = np.reshape(k_depth_vectors_next, new_shape)
    
    def plot_depth_maps(self):
        plt.figure(figsize=(6, 6))
        # Define the color limits based on the overall min and max values in the data
        
        for i in range(self.k):
            plt.clf()  # Clear the current figure
            plt.imshow(self.k_depth_matrices[i], cmap='Blues', vmin=0, vmax=3)
            plt.colorbar(label='Water Depth [m]')
            plt.title(f'prj_{self.prj_num}, plan_{self.plan_num}, cluster_{i}')
            plt.pause(1)  # Pause to update the plot
            
        plt.show()

    def load_tiff_data(self):
        os.chdir(self.terrain_path)
        self.tiff_data = tifffile.imread(self.tiff_name)
        plt.imshow(self.tiff_data, cmap='terrain')
        plt.colorbar()
        plt.title(f'Terrain of prj_{self.prj_num} - plan_{self.plan_num}')
        plt.show()
        
        
    def calculate_num_patches(self):
        # Calculate patch size
        self.meters_in_patch = self.meters_in_cell * self.cells_in_patch # both are even, so meters_in_patch is even
        self.tiff_points_in_patch = self.meters_in_patch + 1  # e.g., for 2m terrain, you need 3 points (edges and middle)

        # Calculate num of patches
        self.num_patches_row = (self.num_rows - 1) // self.cells_in_patch  # -1 is to avoid non-square cells
        self.num_patches_col = (self.num_cols - 1) // self.cells_in_patch  # -1 is to avoid non-square cells
        self.num_patches = self.num_patches_row * self.num_patches_col

        # when looking at dual patches we throw the first cells_in_patch/2 in both dims. from there the same computation should be performed. as the last cell must be discarded in both dims to avoid non square cells.
        self.num_patches_row_dual = (self.num_rows - int(self.cells_in_patch/2) - 1) // self.cells_in_patch  # When taking dual patches, the number of patches diminishes by 1
        self.num_patches_col_dual = (self.num_cols - int(self.cells_in_patch/2) - 1) // self.cells_in_patch
        self.num_patches_dual = self.num_patches_row_dual * self.num_patches_col_dual

    def tiff_patches(self):
        self.patches_tiff = np.zeros((self.num_patches, self.tiff_points_in_patch, self.tiff_points_in_patch))
        self.patches_tiff_dual = np.zeros((self.num_patches_dual, self.tiff_points_in_patch, self.tiff_points_in_patch))


        row_indices = np.arange(0, self.num_patches_row, 1)*(self.tiff_points_in_patch-1) # on a paper, think, suppose each patch has 5 tiff points, then the indices would be [0-5, 5-10, 10-15, 15-20]. so, the indices are [0,5,15,..]. 5 is tiff_points_in_patch - 1
        col_indices =  np.arange(0, self.num_patches_col , 1)*(self.tiff_points_in_patch-1) # just like columns
        rows, cols = np.meshgrid(row_indices, col_indices)
        # ravel() returns a 1-D array  of the input
        # indices for the upper left corner of each tiff patch are returned. we startd from the upper left corner of the big tiff, and return the upper left corners of the patches of the 1st column patches, then the 2nd column patches...
        indices = np.column_stack((rows.ravel(), cols.ravel())) # it provides the indices of patches in a column major manner: [0,0], [10,0], [20,0]

        for i, (row, col) in enumerate(indices):
          patch = self.tiff_data[row:row+self.tiff_points_in_patch, col:col+self.tiff_points_in_patch] # row:row+tiff_points_in_patch 0:11 is 0,1,2,3,4,5,7,8,9,10 (exactly as needed)
          self.patches_tiff[i] = copy.deepcopy(patch)

        # dual
        # meters_in_patch is neccessarily even since it is a multiplication of two even numbers (cells_in_patch and meters_in_cell)
        tiff_strts_from = int(self.meters_in_patch/2) # try it on a piece of paper, if each patch has 10 meters, then you have to throw the first 5 meters. throwing the first 5 meters means throwing: [0,1,2,3,4]. thus means starting from 5 in both dimensions
        self.tiff_data_dual = self.tiff_data[tiff_strts_from:, tiff_strts_from:]

        row_indices = np.arange(0, self.num_patches_row_dual, 1)*(self.tiff_points_in_patch-1) # on a paper, think, suppose each patch has 5 tiff points, then the indices would be [0-5, 5-10, 10-15, 15-20]. so, the indices are [0,5,15,..]. 5 is tiff_points_in_patch - 1
        col_indices =  np.arange(0, self.num_patches_col_dual , 1)*(self.tiff_points_in_patch-1) # just like columns
        rows, cols = np.meshgrid(row_indices, col_indices)
        
        indices = np.column_stack((rows.ravel(), cols.ravel())) # it provides the indices of patches in a column major manner: [0,0], [10,0], [20,0], [30,0]

        for i, (row, col) in enumerate(indices):
          patch = self.tiff_data_dual[row:row+self.tiff_points_in_patch, col:col+self.tiff_points_in_patch] # row:row+tiff_points_in_patch 0:6 is 0,1,2,3,4,5 (exactly as needed)
          self.patches_tiff_dual[i] = copy.deepcopy(patch)
               
    def depth_patches(self, cluster):
        if (cluster % 2 == 0):
            self.patches_depth = np.zeros((self.num_patches, self.cells_in_patch, self.cells_in_patch))
            self.patches_depth_next = np.zeros((self.num_patches, self.cells_in_patch, self.cells_in_patch))
    
    
            row_indices = np.arange(0, self.num_patches_row, 1)*self.cells_in_patch # on a paper think, suppose each patch holds 10 cells, then the indices would be [0-9, 10-19, 20,29..]. so the indices are [0, 10, 20]. 10 is cells_in_patch
            col_indices =  np.arange(0, self.num_patches_col, 1)*self.cells_in_patch
            rows, cols = np.meshgrid(row_indices, col_indices)
            indices = np.column_stack((rows.ravel(), cols.ravel()))
    
            for i, (row, col) in enumerate(indices):
                patch = self.k_depth_matrices[cluster, row:row+self.cells_in_patch, col:col+self.cells_in_patch] # everything is for one k only
                patch_next = self.k_depth_matrices_next[cluster, row:row+self.cells_in_patch, col:col+self.cells_in_patch]  # everything is for one k only
    
                self.patches_depth[i] =  copy.deepcopy(patch)
                self.patches_depth_next[i] =  copy.deepcopy(patch_next)
                
           
            # Plot the patches
            fig, axes = plt.subplots(self.num_patches_row, self.num_patches_col, figsize=(12, 8))
            for i, ax in enumerate(axes.T.flat):  # Adjust to column-major order using axes.T.flat
                if i < self.num_patches:
                    ax.imshow(self.patches_depth[i], cmap='Blues', vmin=0, vmax=3)
                ax.axis('off')
                
    
            plt.tight_layout()
            plt.show()
             
        else:
            self.patches_depth_dual = np.zeros((self.num_patches_dual, self.cells_in_patch, self.cells_in_patch))
            self.patches_depth_next_dual = np.zeros((self.num_patches_dual, self.cells_in_patch, self.cells_in_patch))

            depth_strts_from = int(self.cells_in_patch/2) # try it on a piece of paper, if each patch has 10 cells, then you have to throw the first 5 cells. throwing the first 5 cells means throwing: [0,1,2,3,4]. this means starting from 5 in both dimensions
            k_depth_matrices_dual = self.k_depth_matrices[:, depth_strts_from:, depth_strts_from:]
            k_depth_matrices_next_dual = self.k_depth_matrices_next[:, depth_strts_from:, depth_strts_from:]
    
            row_indices = np.arange(0, self.num_patches_row_dual, 1) * self.cells_in_patch # on a paper think, suppose each patch holds 10 cells, then the indices would be [0-9, 10-19, 20,29..]. so the indices are [0, 10, 20]. 10 is cells_in_patch
            col_indices =  np.arange(0, self.num_patches_col_dual, 1) * self.cells_in_patch
            rows, cols = np.meshgrid(row_indices, col_indices)
            indices = np.column_stack((rows.ravel(), cols.ravel()))
    
            for i, (row, col) in enumerate(indices):
                patch = k_depth_matrices_dual[cluster, row:row+self.cells_in_patch, col:col+self.cells_in_patch] # everything is for one k only
                patch_next = k_depth_matrices_next_dual[cluster, row:row+self.cells_in_patch, col:col+self.cells_in_patch]  # everything is for one k only
    
                self.patches_depth_dual[i] =  copy.deepcopy(patch)
                self.patches_depth_next_dual[i] =  copy.deepcopy(patch_next)
            
            # Plot the patches
            fig, axes = plt.subplots(self.num_patches_row_dual, self.num_patches_col_dual, figsize=(12, 8))
            for i, ax in enumerate(axes.T.flat):  # Adjust to column-major order using axes.T.flat
                if i < self.num_patches_dual:
                    ax.imshow(self.patches_depth_dual[i], cmap='Blues', vmin=0, vmax=3)
                ax.axis('off')
                
    
            plt.tight_layout()
            plt.show()

    def add_samples(self, patches_depth, patches_depth_next, patches_tiff): # the method takes 5 lists
        
        # Elevate the patches by the sampled value        
        patches_tiff = copy.deepcopy(patches_tiff)
        patches_depth = copy.deepcopy(patches_depth)
        patches_depth_next = copy.deepcopy(patches_depth_next)
        
        
        # Plot the patches
        if self.cluster_counter%2 ==0: 
            x = self.num_patches_row
            y = self.num_patches_col
            num = self.num_patches
        else:
            x = self.num_patches_row_dual
            y = self.num_patches_col_dual
            num = self.num_patches_dual

        fig, axes = plt.subplots(x, y, figsize=(12, 8))
        for i, ax in enumerate(axes.T.flat):  # Adjust to column-major order using axes.T.flat
            if i < num:
                ax.imshow(patches_depth[i], cmap='Blues', vmin=0, vmax=3)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        
        fig, axes = plt.subplots(x, y, figsize=(12, 8))
        for i, ax in enumerate(axes.T.flat):  # Adjust to column-major order using axes.T.flat
            if i < num:
                ax.imshow(patches_tiff[i], cmap = 'terrain', vmin=np.min(patches_tiff), vmax=np.max(patches_tiff))
            ax.axis('off')
        plt.tight_layout()
        plt.show()
        
            # Increment cluster counter for next call
        self.cluster_counter += 1


        for patch_depth, patch_depth_next, patch_terrain in zip(patches_depth, patches_depth_next, patches_tiff): # the mehod uses zip to iteratre over the 5 lists 
                sample = {'terrain': patch_terrain,
                          'depth': patch_depth,
                          'depth_next': patch_depth_next}
                
                for key, value in sample.items():
                    self.database[key].append(value)

    def flip(self, dual, side):
        if side == 'horizontal':
            if dual==False:
                self.patches_depth = np.fliplr(self.patches_depth) # equivalent to np.flip(axis=1)
                self.patches_depth_next = np.fliplr(self.patches_depth_next)
                
                patches_tiff_copy = copy.deepcopy(self.patches_tiff)
                patches_tiff_flipped = np.fliplr(patches_tiff_copy)

            else: 
                self.patches_depth_dual = np.fliplr(self.patches_depth_dual)
                self.patches_depth_next_dual = np.fliplr(self.patches_depth_next_dual)
                
                patches_tiff_copy = copy.deepcopy(self.patches_tiff_dual)
                patches_tiff_flipped = np.fliplr(patches_tiff_copy)                
        
        if side == 'vertical' :
            if dual==False:
                self.patches_depth = np.flip(self.patches_depth, axis=2) # stupid me, didn't think i need to specify the axis.
                self.patches_depth_next = np.flip(self.patches_depth_next, axis=2) # stupid me, didn't think i need to specify the axis.
                
                patches_tiff_copy = copy.deepcopy(self.patches_tiff)
                patches_tiff_flipped = np.flip(patches_tiff_copy, axis=2) # stupid me, didn't think i need to specify the axis.
            else: 
                self.patches_depth_dual = np.flip(self.patches_depth_dual, axis=2) # stupid me, didn't think i need to specify the axis.
                self.patches_depth_next_dual = np.flip(self.patches_depth_next_dual, axis=2) # stupid me, didn't think i need to specify the axis.

                patches_tiff_copy = copy.deepcopy(self.patches_tiff_dual)
                patches_tiff_flipped = np.flip(patches_tiff_copy, axis=2) # stupid me, didn't think i need to specify the axis.
                
        return patches_tiff_flipped
    
    def rotate(self, patches_tiff_flipped, dual, angle): #rotation comes after flipping, thus flipped tiffs must be provided 
        if dual==False:
            self.patches_depth = ndimage.rotate(self.patches_depth, angle, reshape=False, axes = (1,2)) # stupid me, didn't think i need to specify the axes
            self.patches_depth_next = ndimage.rotate(self.patches_depth_next, angle, reshape=False, axes = (1,2)) # stupid me, didn't think i need to specify the axes
            patches_tiff_rotated = ndimage.rotate(patches_tiff_flipped, angle, reshape=False, axes = (1,2)) # stupid me, didn't think i need to specify the axes
       
        else: 
            self.patches_depth_dual = ndimage.rotate(self.patches_depth_dual, angle, reshape=False, axes = (1,2)) # stupid me, didn't think i need to specify the axes
            self.patches_depth_next_dual = ndimage.rotate(self.patches_depth_next_dual, angle, reshape=False, axes = (1,2))# stupid me, didn't think i need to specify the axes
            patches_tiff_rotated = ndimage.rotate(patches_tiff_flipped, angle, reshape=False, axes = (1,2)) # stupid me, didn't think i need to specify the axes
            
        return patches_tiff_rotated
    
     
    def populate_from_all_clusters(self):
        print('Now in prj_' + self.prj_num + ' ' + 'plan_' + self.plan_num)
        for j in [0, 1, 2, 3]: 
            self.depth_patches(cluster = j) #this is a method that update the depth_patches
    
            if j==0: 
                self.add_samples(self.patches_depth, self.patches_depth_next, self.patches_tiff)

            if j==1: 
                patches_tiff_flipped = self.flip(dual=True, side = 'horizontal')
                patches_tiff_rotated = self.rotate(patches_tiff_flipped, dual=True, angle = 90)
                self.add_samples(self.patches_depth_dual, self.patches_depth_next_dual, patches_tiff_rotated)                       

            if j==2: 
                patches_tiff_flipped = self.flip(dual=False, side = 'vertical')
                patches_tiff_rotated = self.rotate(patches_tiff_flipped, dual=False, angle = 270)
                self.add_samples(self.patches_depth, self.patches_depth_next, patches_tiff_rotated)

            if j==3: 
                patches_tiff_flipped = self.flip(dual=True, side = 'vertical')
                patches_tiff_rotated = self.rotate(patches_tiff_flipped, dual=True, angle = 180)
                self.add_samples(self.patches_depth_dual, self.patches_depth_next_dual, patches_tiff_rotated)

            print('cluster ' + str(j) + ' is done')
            
    def delete_dry_patches(self): # and very high depths
        indices_to_remove = []
        
        # Iterate through depth and depth_next matrices
        for i in range(len(self.database['depth'])):
            depth_sum = np.sum(self.database['depth'][i])
            depth_next_sum = np.sum(self.database['depth_next'][i])

            
            ################################################################################################################ 10
            # Check if the sum of elements in depth and depth_next matrices is smaller than 10^-3
            if (depth_sum + depth_next_sum) < 10: # must rethink this 
                indices_to_remove.append(i)
                

        # Remove corresponding samples
        for key in self.database.keys():
            self.database[key] = [sample for i, sample in enumerate(self.database[key]) if i not in indices_to_remove]


    def run_all_methods(self):
        self.populate_from_HDF()
        self.cluster_via_KMeans()
        self.find_num_rows_cols_in_HECRAS()
        self.k_matrices_depth_WSE()
        self.plot_depth_maps()
        self.load_tiff_data()
        self.calculate_num_patches()
        self.tiff_patches()
        self.populate_from_all_clusters()
        self.delete_dry_patches()
#%%
# Instantiate the class and run methods for the specific project, plan, and cluster
simulation = from_HDF(prj_num='03', prj_name='hecras_on_03', plan_num='11')
simulation.run_all_methods()
