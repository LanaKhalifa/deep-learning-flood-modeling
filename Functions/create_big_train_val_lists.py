import pickle
import os
import numpy as np
#%% prj 03
prj_num = '03'
directory = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Database/{prj_num}/'

def concatenate_and_save(file_prefix, output_filename):
    all_items = []

    # Loop through the specified range of pickle files
    for i in range(5):  # leave last index for test set
        file_path = os.path.join(directory, f'{file_prefix}_{i}.pkl')
        with open(file_path, 'rb') as file:
            items = pickle.load(file)
            # Convert each item to float32 and append to the list
            for item in items:
                if isinstance(item, np.ndarray):
                    all_items.append(item.astype(np.float32))
                else:
                    all_items.append(item)

    # Save all_items to a new pickle file
    output_file_path = os.path.join(directory, output_filename)
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(all_items, output_file)

    print(f"All {file_prefix} items have been saved to {output_file_path}")
    return all_items

# Concatenate and save depths, depths_next, and terrains
all_depths = concatenate_and_save('depths', 'big_train_val_depths.pkl')
all_depths_next = concatenate_and_save('depths_next', 'big_train_val_depths_next.pkl')
all_terrains = concatenate_and_save('terrains', 'big_train_val_terrains.pkl')

# Ensure the lengths of the three lists are identical
assert len(all_depths) == len(all_depths_next) == len(all_terrains), "The lengths of the lists are not identical."

# Ensure every element in all_depths is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths), "Not all items in all_depths are 32 by 32."

# Ensure every element in all_depths_next is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths_next), "Not all items in all_depths_next are 32 by 32."

# Ensure every element in all_terrains is 321 by 321
assert all(isinstance(item, np.ndarray) and item.shape == (321, 321) for item in all_terrains), "Not all items in all_terrains are 321 by 321."

print("All checks passed successfully for prj 03.")
#%% prj 04
prj_num = '04'
directory = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Database/{prj_num}/'

def concatenate_and_save(file_prefix, output_filename):
    all_items = []

    # Loop through the specified range of pickle files
    for i in range(5):  # leave last index for test set
        file_path = os.path.join(directory, f'{file_prefix}_{i}.pkl')
        with open(file_path, 'rb') as file:
            items = pickle.load(file)
            # Convert each item to float32 and append to the list
            for item in items:
                if isinstance(item, np.ndarray):
                    all_items.append(item.astype(np.float32))
                else:
                    all_items.append(item)

    # Save all_items to a new pickle file
    output_file_path = os.path.join(directory, output_filename)
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(all_items, output_file)

    print(f"All {file_prefix} items have been saved to {output_file_path}")
    return all_items

# Concatenate and save depths, depths_next, and terrains
all_depths = concatenate_and_save('depths', 'big_train_val_depths.pkl')
all_depths_next = concatenate_and_save('depths_next', 'big_train_val_depths_next.pkl')
all_terrains = concatenate_and_save('terrains', 'big_train_val_terrains.pkl')

# Ensure the lengths of the three lists are identical
assert len(all_depths) == len(all_depths_next) == len(all_terrains), "The lengths of the lists are not identical."

# Ensure every element in all_depths is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths), "Not all items in all_depths are 32 by 32."

# Ensure every element in all_depths_next is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths_next), "Not all items in all_depths_next are 32 by 32."

# Ensure every element in all_terrains is 321 by 321
assert all(isinstance(item, np.ndarray) and item.shape == (321, 321) for item in all_terrains), "Not all items in all_terrains are 321 by 321."

print("All checks passed successfully for prj 04.")
#%% prj 05
prj_num = '05'
directory = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Database/{prj_num}/'

def concatenate_and_save(file_prefix, output_filename):
    all_items = []

    # Loop through the specified range of pickle files
    for i in range(12):  # leave last index for test set
        file_path = os.path.join(directory, f'{file_prefix}_{i}.pkl')
        with open(file_path, 'rb') as file:
            items = pickle.load(file)
            # Convert each item to float32 and append to the list
            for item in items:
                if isinstance(item, np.ndarray):
                    all_items.append(item.astype(np.float32))
                else:
                    all_items.append(item)

    # Save all_items to a new pickle file
    output_file_path = os.path.join(directory, output_filename)
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(all_items, output_file)

    print(f"All {file_prefix} items have been saved to {output_file_path}")
    return all_items

# Concatenate and save depths, depths_next, and terrains
all_depths = concatenate_and_save('depths', 'big_train_val_depths.pkl')
all_depths_next = concatenate_and_save('depths_next', 'big_train_val_depths_next.pkl')
all_terrains = concatenate_and_save('terrains', 'big_train_val_terrains.pkl')

# Ensure the lengths of the three lists are identical
assert len(all_depths) == len(all_depths_next) == len(all_terrains), "The lengths of the lists are not identical."

# Ensure every element in all_depths is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths), "Not all items in all_depths are 32 by 32."

# Ensure every element in all_depths_next is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths_next), "Not all items in all_depths_next are 32 by 32."

# Ensure every element in all_terrains is 321 by 321
assert all(isinstance(item, np.ndarray) and item.shape == (321, 321) for item in all_terrains), "Not all items in all_terrains are 321 by 321."

print("All checks passed successfully for prj 05.")
#%% prj 06
prj_num = '06'
directory = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Database/{prj_num}/'

def concatenate_and_save(file_prefix, output_filename):
    all_items = []

    # Loop through the specified range of pickle files
    for i in range(6):  # leave last index for test set
        file_path = os.path.join(directory, f'{file_prefix}_{i}.pkl')
        with open(file_path, 'rb') as file:
            items = pickle.load(file)
            # Convert each item to float32 and append to the list
            for item in items:
                if isinstance(item, np.ndarray):
                    all_items.append(item.astype(np.float32))
                else:
                    all_items.append(item)

    # Save all_items to a new pickle file
    output_file_path = os.path.join(directory, output_filename)
    with open(output_file_path, 'wb') as output_file:
        pickle.dump(all_items, output_file)

    print(f"All {file_prefix} items have been saved to {output_file_path}")
    return all_items

# Concatenate and save depths, depths_next, and terrains
all_depths = concatenate_and_save('depths', 'big_train_val_depths.pkl')
all_depths_next = concatenate_and_save('depths_next', 'big_train_val_depths_next.pkl')
all_terrains = concatenate_and_save('terrains', 'big_train_val_terrains.pkl')

# Ensure the lengths of the three lists are identical
assert len(all_depths) == len(all_depths_next) == len(all_terrains), "The lengths of the lists are not identical."

# Ensure every element in all_depths is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths), "Not all items in all_depths are 32 by 32."

# Ensure every element in all_depths_next is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths_next), "Not all items in all_depths_next are 32 by 32."

# Ensure every element in all_terrains is 321 by 321
assert all(isinstance(item, np.ndarray) and item.shape == (321, 321) for item in all_terrains), "Not all items in all_terrains are 321 by 321."

print("All checks passed successfully for prj 06.")

#%% save a big list from all prjs
# Define the directory where the final combined files will be saved
combined_directory = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Database'

# Initialize empty lists to store the final combined data
big_train_val_depths = []
big_train_val_depths_next = []
big_train_val_terrains = []

# List of project numbers whose concatenated files need to be combined
projects = ['03', '04', '05', '06']

# Iterate over all projects and load the combined 'big_train_val_depths.pkl', 'big_train_val_depths_next.pkl', and 'big_train_val_terrains.pkl' files for each
for prj_num in projects:
    directory = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Database/{prj_num}/'
    
    # Load and combine depths
    depths_path = os.path.join(directory, 'big_train_val_depths.pkl')
    with open(depths_path, 'rb') as file:
        project_depths = pickle.load(file)
        big_train_val_depths.extend(project_depths)
    
    # Load and combine depths_next
    depths_next_path = os.path.join(directory, 'big_train_val_depths_next.pkl')
    with open(depths_next_path, 'rb') as file:
        project_depths_next = pickle.load(file)
        big_train_val_depths_next.extend(project_depths_next)
    
    # Load and combine terrains
    terrains_path = os.path.join(directory, 'big_train_val_terrains.pkl')
    with open(terrains_path, 'rb') as file:
        project_terrains = pickle.load(file)
        big_train_val_terrains.extend(project_terrains)

# Save the final combined lists to new pickle files
depths_output_path = os.path.join(combined_directory, 'big_train_val_depths.pkl')
depths_next_output_path = os.path.join(combined_directory, 'big_train_val_depths_next.pkl')
terrains_output_path = os.path.join(combined_directory, 'big_train_val_terrains.pkl')

with open(depths_output_path, 'wb') as output_file:
    pickle.dump(big_train_val_depths, output_file)
    
with open(depths_next_output_path, 'wb') as output_file:
    pickle.dump(big_train_val_depths_next, output_file)
    
with open(terrains_output_path, 'wb') as output_file:
    pickle.dump(big_train_val_terrains, output_file)

print(f"Final combined big_train_val_depths has been saved to {depths_output_path}")
print(f"Final combined big_train_val_depths_next has been saved to {depths_next_output_path}")
print(f"Final combined big_train_val_terrains has been saved to {terrains_output_path}")

# Verify the combined data
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in big_train_val_depths), "Not all items in big_train_val_depths are 32 by 32."
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in big_train_val_depths_next), "Not all items in big_train_val_depths_next are 32 by 32."
assert all(isinstance(item, np.ndarray) and item.shape == (321, 321) for item in big_train_val_terrains), "Not all items in big_train_val_terrains are 321 by 321."

print("All checks passed successfully for the final combined lists.")

