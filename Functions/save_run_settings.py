import os
import pandas as pd

def save_run_settings(base_dir, prj_num, Architecture_num, trial_num, settings):
    # Define the path to the CSV file
    settings_path = os.path.join(base_dir, f'{Architecture_num}/run_settings.csv')
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    
    # Check if the CSV file exists
    if os.path.exists(settings_path):
        # Read the existing CSV file
        df = pd.read_csv(settings_path, index_col=0)
    else:
        # Create a new DataFrame if the CSV file doesn't exist
        df = pd.DataFrame(columns=[
            'trial_num', 'prj_num', 'BC_thickness', 'num_epochs', 'weight_init', 'initial_lr',
            'terrain_c', 'down_c_start', 'down_c1', 'down_c2', 'down_c_end', 'down_act', 'down_type',
            'arch_num_layers', 'arch_num_attentions', 'arch_num_c', 'input_c', 'arch_input_c', 'arch_act', 'arch_last_act'
        ])
    
    # Update the DataFrame with the new settings for the current trial_num
    df.loc[trial_num] = [
        trial_num, prj_num, settings['BC_thickness'], settings['num_epochs'], settings['weight_init'], settings['initial_lr'],
        settings['terrain_c'], settings['down_c_start'], settings['down_c1'], settings['down_c2'], settings['down_c_end'], settings['down_act'], settings['down_type'],
        settings['arch_num_layers'], settings['arch_num_attentions'], settings['arch_num_c'], settings['input_c'], settings['arch_input_c'], settings['arch_act'], settings['arch_last_act']
    ]
    
    # Save the DataFrame back to the CSV file
    df.to_csv(settings_path)