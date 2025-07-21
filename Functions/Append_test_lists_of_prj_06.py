import pickle
import os
import numpy as np

prj_num = '06'
directory = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Database/{prj_num}/'

def concatenate_and_save(file_prefix, output_filename):
    all_items = []

    # Loop through the specified range of pickle files
    i =5
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
all_depths = concatenate_and_save('depths', 'test_depths.pkl')
all_depths_next = concatenate_and_save('depths_next', 'test_depths_next.pkl')
all_terrains = concatenate_and_save('terrains', 'test_terrains.pkl')

#%%
# Ensure the lengths of the three lists are identical
assert len(all_depths) == len(all_depths_next) == len(all_terrains), "The lengths of the lists are not identical."

# Ensure every element in all_depths is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths), "Not all items in all_depths are 32 by 32."

# Ensure every element in all_depths_next is 32 by 32
assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths_next), "Not all items in all_depths_next are 32 by 32."

# Ensure every element in all_terrains is 321 by 321
assert all(isinstance(item, np.ndarray) and item.shape == (321, 321) for item in all_terrains), "Not all items in all_terrains are 321 by 321."

print("All checks passed successfully.")
