import os
import pandas as pd

def save_val_quartiles(base_dir, Architecture_num, trial_num, Q1, Q2, Q3, avg_netG_val_loss, std_netG_val_loss, avg_dummy_val_loss, std_dummy_val_loss):
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
        df = pd.DataFrame(columns=['trial_num', 'Q1', 'Q2', 'Q3', 'avg_netG_val_loss', 'std_netG_val_loss', 'avg_dummy_val_loss', 'std_dummy_val_loss'])
    
    # Update the DataFrame with the new quartiles for the current trial_num
    df.loc[trial_num] = [trial_num, Q1, Q2, Q3, avg_netG_val_loss, std_netG_val_loss, avg_dummy_val_loss, std_dummy_val_loss]
    
    # Save the DataFrame back to the CSV file
    df.to_csv(quartiles_path)

