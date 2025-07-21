import os
import pandas as pd

def save_quartiles(base_dir, Architecture_num, trial_num, Q1, Q2, Q3, min_val, max_val, which):
    # Define the path to the CSV file
    quartiles_path = os.path.join(base_dir, f'{Architecture_num}/{trial_num}/boxplot/rae_quartiles_{which}.csv')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(quartiles_path), exist_ok=True)
    
    # Check if the CSV file exists
    if os.path.exists(quartiles_path):
        # Read the existing CSV file
        df = pd.read_csv(quartiles_path, index_col=0)
    else:
        # Create a new DataFrame if the CSV file doesn't exist
        df = pd.DataFrame(columns=['trial_num', 'Q1', 'Q2', 'Q3', 'min_val', 'max_val'])
    
    # Update the DataFrame with the new quartiles for the current trial_num
    df.loc[trial_num] = [trial_num, Q1, Q2, Q3, min_val, max_val]
    
    # Save the DataFrame back to the CSV file
    df.to_csv(quartiles_path)
