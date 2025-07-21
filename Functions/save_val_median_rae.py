import os
import pandas as pd


def save_val_quartiles(base_dir, Architecture_num, trial_num, Q1, Q2, Q3):
    # Define the path to the CSV file
    quartiles_path = os.path.join(base_dir, f'{Architecture_num}/rae_quartiles.csv')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(quartiles_path), exist_ok=True)
    
    # Check if the CSV file exists
    if os.path.exists(quartiles_path):
        # Read the existing CSV file
        df = pd.read_csv(quartiles_path, index_col=0)
    else:
        # Create a new DataFrame if the CSV file doesn't exist
        df = pd.DataFrame(columns=['trial_num', 'Q1', 'Q2', 'Q3'])
    
    # Update the DataFrame with the new quartiles for the current trial_num
    df.loc[trial_num] = [trial_num, Q1, Q2, Q3]
    
    # Save the DataFrame back to the CSV file
    df.to_csv(quartiles_path)

# Example usage:
base_dir = '/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF'
Architecture_num = 'example_architecture'  # Replace with your actual architecture number
trial_num = 'example_trial'  # Replace with your actual trial number
Q1, Q2, Q3 = 0.25, 0.50, 0.75  # Replace with the actual quartile values

save_val_quartiles(base_dir, Architecture_num, trial_num, Q1, Q2, Q3)


if False:
    def save_val_median_rae(base_dir, Architecture_num, trial_num, val_median_rae):
        # Define the path to the CSV file
        rae_median_path = os.path.join(base_dir, f'{Architecture_num}/rae_median.csv')
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(rae_median_path), exist_ok=True)
        
        # Check if the CSV file exists
        if os.path.exists(rae_median_path):
            # Read the existing CSV file
            df = pd.read_csv(rae_median_path, index_col=0)
        else:
            # Create a new DataFrame if the CSV file doesn't exist
            df = pd.DataFrame(columns=['trial_num', 'val_median_rae'])
        
        # Update the DataFrame with the new val_median_rae for the current trial_num
        df.loc[trial_num] = [trial_num, val_median_rae]
        
        # Save the DataFrame back to the CSV file
        df.to_csv(rae_median_path)
