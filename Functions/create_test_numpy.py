import pickle
import os

# List of project numbers
project_numbers = ['03', '04', '05'] # '06' is excluded since i forgot it 

# Initialize lists to hold the concatenated results
huge_depths = []
huge_depths_next = []
huge_terrains = []

# Function to load and concatenate data from pickle files
def load_and_concatenate(prj_num, file_name, huge_list):
    directory = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Database/{prj_num}/'
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        huge_list.extend(data)
    return huge_list

# Load and concatenate data from all project directories
for prj_num in project_numbers:
    huge_depths = load_and_concatenate(prj_num, 'test_depths.pkl', huge_depths)
    huge_depths_next = load_and_concatenate(prj_num, 'test_depths_next.pkl', huge_depths_next)
    huge_terrains = load_and_concatenate(prj_num, 'test_terrains.pkl', huge_terrains)

if True:
    # Save the concatenated lists to new pickle files
    output_directory = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/Database/'
    
    with open(os.path.join(output_directory, 'test_depths.pkl'), 'wb') as output_file:
        pickle.dump(huge_depths, output_file)
    
    with open(os.path.join(output_directory, 'test_depths_next.pkl'), 'wb') as output_file:
        pickle.dump(huge_depths_next, output_file)
    
    with open(os.path.join(output_directory, 'test_terrains.pkl'), 'wb') as output_file:
        pickle.dump(huge_terrains, output_file)
    
    print("All huge lists have been saved successfully.")

