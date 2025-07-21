import pickle

def load_list_from_file(prj_num, var_name, curr_delta_t, spatial):
    """
    Load a list from a file using pickle.
    
    Parameters:
    - prj_num: Project number.
    - var_name: Variable name.
    - cluster: Cluster index.
    - curr_delta_t: Delta time (default '60mins_').
    - spatial: Spatial dimension (default '32').
    
    Returns:
    - The loaded list.
    """
    file_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF/complete_dataset_lists_from_prjs/{prj_num}/{var_name}_delta_t_{curr_delta_t}XY_{spatial}.pkl'
    print(f'Loading file from: {file_path}')
    with open(file_path, 'rb') as f:
        lst = pickle.load(f)
    return lst
