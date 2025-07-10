# This script processes HEC-RAS simulation outputs into patch data for deep learning.
# Each number in the sublists corresponds to a plan ID (plan_num) — a simulation run in HEC-RAS.
# For memory efficiency, we process simulations in groups (sublists) using PatchExtractorProcessor.
# Output patches are saved per group under ./Database/saved_in_chunks/{prj_num}/.
# Then all training/validation chunks are concatenated across all projects (excluding test groups).

import os
import pickle
import numpy as np
import logging
from PatchExtractorProcessor import PatchExtractorProcessor  # Update with actual class name if different

logging.basicConfig(level=logging.INFO)

prj_03_sublists = [
    ['04', '05', '06', '07', '08', '09', '10'],
    ['13', '14', '15', '16', '17', '18', '19'],
    ['20', '21', '22', '23', '24', '25', '26'],
    ['31', '32', '33', '35', '38', '39', '40'],
    ['51', '52', '53', '55', '56', '57', '58'],
    ['61', '63', '64', '65', '71', '72', '73']]

prj_04_sublists = [
    ['01', '02', '03', '04', '05', '06', '07'],
    ['08', '10', '12', '13', '14', '15', '16'],
    ['17', '18', '20', '21', '22', '23', '24'],
    ['25', '26', '27', '28', '29', '30', '31'],
    ['34', '35', '37'],
    ['38', '39', '40', '41', '42', '44', '45']]

prj_05_sublists = [
    ['01', '02', '03', '04', '05', '06', '07'],
    ['08', '10', '11', '12', '13', '14', '15'],
    ['16', '17', '18', '19', '20', '21', '22'],
    ['23', '24', '25', '26', '27', '28', '29'],
    ['30', '31', '32', '35', '36', '37', '38'],
    ['39', '40', '43', '44', '45', '46', '47'],
    ['48', '49', '50', '51', '52', '53', '54'],
    ['55', '56', '57', '58', '59', '60', '61'],
    ['62', '63', '64', '65', '66', '67', '68'],
    ['69', '70', '71', '72', '73', '74', '75'],
    ['76', '77', '78', '79', '80', '81', '82'],
    ['83'],
    ['84', '85', '86', '87', '88', '89', '90']]

prj_06_sublists = [
    ['01', '02', '03', '04', '05', '06', '07'],
    ['08', '09', '10', '11', '12', '16', '17'],
    ['18', '19', '20', '21', '22', '23', '26'],
    ['27', '28', '29', '30', '31', '32', '33'],
    ['34', '35', '36', '37', '38', '39', '40'],
    ['41', '42', '43'],
    ['44', '45', '46', '47', '48', '49', '50']]

prj_sublists = {
    '03': ('hecras_on_03', prj_03_sublists),
    '04': ('HECRAS', prj_04_sublists),
    '05': ('HECRAS', prj_05_sublists),
    '06': ('HECRAS', prj_06_sublists),}

def process_project(prj_num, prj_name, sublists):
    for idx, plan_nums in enumerate(sublists):
        logging.info(f"Processing sublist {idx} of project {prj_num}")

        instances = [PatchExtractorProcessor(prj_num, prj_name, plan_num) for plan_num in plan_nums]
        for instance in instances:
            instance.generate_patches()

        terrains = np.concatenate([inst.database['terrain'] for inst in instances if len(inst.database['depth']) > 0], axis=0)
        depths = np.concatenate([inst.database['depth'] for inst in instances if len(inst.database['depth']) > 0], axis=0)
        depths_next = np.concatenate([inst.database['depth_next'] for inst in instances if len(inst.database['depth_next']) > 0], axis=0)

        save_dir = f'./Database/saved_in_chunks/{prj_num}/'
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, f'terrains_sublist_{idx}.pkl'), 'wb') as f:
            pickle.dump(terrains, f)
        with open(os.path.join(save_dir, f'depths_sublist_{idx}.pkl'), 'wb') as f:
            pickle.dump(depths, f)
        with open(os.path.join(save_dir, f'depths_next_sublist_{idx}.pkl'), 'wb') as f:
            pickle.dump(depths_next, f)

        logging.info(f"Sublist {idx} saved successfully.")

def concatenate_all_train_val_chunks(prj_sublists):
    all_depths, all_depths_next, all_terrains = [], [], []

    for prj_num, (_, sublists) in prj_sublists.items():
        save_dir = f'./Database/saved_in_chunks/{prj_num}/'
        last_index = len(sublists) - 1

        for idx in range(last_index):  # exclude test set
            with open(os.path.join(save_dir, f'depths_sublist_{idx}.pkl'), 'rb') as f:
                all_depths.extend(pickle.load(f))
            with open(os.path.join(save_dir, f'depths_next_sublist_{idx}.pkl'), 'rb') as f:
                all_depths_next.extend(pickle.load(f))
            with open(os.path.join(save_dir, f'terrains_sublist_{idx}.pkl'), 'rb') as f:
                all_terrains.extend(pickle.load(f))

    output_dir = './Database/'
    with open(os.path.join(output_dir, 'big_train_val_depths.pkl'), 'wb') as f:
        pickle.dump(all_depths, f)
    with open(os.path.join(output_dir, 'big_train_val_depths_next.pkl'), 'wb') as f:
        pickle.dump(all_depths_next, f)
    with open(os.path.join(output_dir, 'big_train_val_terrains.pkl'), 'wb') as f:
        pickle.dump(all_terrains, f)

    logging.info("Final big_train_val files saved across all projects.")

    # Verification
    assert len(all_depths) == len(all_depths_next) == len(all_terrains), "Mismatch in number of samples across lists."
    assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths), "Invalid shape in depths."
    assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths_next), "Invalid shape in depths_next."
    assert all(isinstance(item, np.ndarray) and item.shape == (321, 321) for item in all_terrains), "Invalid shape in terrains."
    logging.info("All checks passed successfully for final combined lists.")

def concatenate_all_test_chunks(prj_sublists):
    all_depths, all_depths_next, all_terrains = [], [], []

    for prj_num, (_, sublists) in prj_sublists.items():
        save_dir = f'./Database/saved_in_chunks/{prj_num}/'
        last_index = len(sublists) - 1  # only use last index for test set

        with open(os.path.join(save_dir, f'depths_sublist_{last_index}.pkl'), 'rb') as f:
            all_depths.extend(pickle.load(f))
        with open(os.path.join(save_dir, f'depths_next_sublist_{last_index}.pkl'), 'rb') as f:
            all_depths_next.extend(pickle.load(f))
        with open(os.path.join(save_dir, f'terrains_sublist_{last_index}.pkl'), 'rb') as f:
            all_terrains.extend(pickle.load(f))

    output_dir = './Database/'
    with open(os.path.join(output_dir, 'big_test_depths.pkl'), 'wb') as f:
        pickle.dump(all_depths, f)
    with open(os.path.join(output_dir, 'big_test_depths_next.pkl'), 'wb') as f:
        pickle.dump(all_depths_next, f)
    with open(os.path.join(output_dir, 'big_test_terrains.pkl'), 'wb') as f:
        pickle.dump(all_terrains, f)

    logging.info("Final big_test files saved across all projects.")

    # (Add assertions or inspection code if needed)

def concatenate_prj03_split():
    prj_num = '03'
    save_dir = f'./Database/saved_in_chunks/{prj_num}/'
    sublists = prj_sublists[prj_num][1]

    train_val_depths, train_val_depths_next, train_val_terrains = [], [], []
    test_depths, test_depths_next, test_terrains = [], [], []

    for idx in range(len(sublists)):
        with open(os.path.join(save_dir, f'depths_sublist_{idx}.pkl'), 'rb') as f:
            depths = pickle.load(f)
        with open(os.path.join(save_dir, f'depths_next_sublist_{idx}.pkl'), 'rb') as f:
            depths_next = pickle.load(f)
        with open(os.path.join(save_dir, f'terrains_sublist_{idx}.pkl'), 'rb') as f:
            terrains = pickle.load(f)

        if idx == len(sublists) - 1:
            test_depths.extend(depths)
            test_depths_next.extend(depths_next)
            test_terrains.extend(terrains)
        else:
            train_val_depths.extend(depths)
            train_val_depths_next.extend(depths_next)
            train_val_terrains.extend(terrains)

    output_dir = './Database/'
    with open(os.path.join(output_dir, 'prj_03_train_val_depths.pkl'), 'wb') as f:
        pickle.dump(train_val_depths, f)
    with open(os.path.join(output_dir, 'prj_03_train_val_depths_next.pkl'), 'wb') as f:
        pickle.dump(train_val_depths_next, f)
    with open(os.path.join(output_dir, 'prj_03_train_val_terrains.pkl'), 'wb') as f:
        pickle.dump(train_val_terrains, f)

    with open(os.path.join(output_dir, 'prj_03_test_depths.pkl'), 'wb') as f:
        pickle.dump(test_depths, f)
    with open(os.path.join(output_dir, 'prj_03_test_depths_next.pkl'), 'wb') as f:
        pickle.dump(test_depths_next, f)
    with open(os.path.join(output_dir, 'prj_03_test_terrains.pkl'), 'wb') as f:
        pickle.dump(test_terrains, f)

    logging.info("prj_03 train/val and test files saved.")

def generate_small_train_val_subset():
    prj_num = '03'
    sublists = prj_sublists[prj_num][1]
    save_dir = f'./Database/saved_in_chunks/{prj_num}/'
    output_dir = './Database/'

    all_depths, all_depths_next, all_terrains = [], [], []

    for idx in range(3):  # only first 3 sublists (21 simulations)
        with open(os.path.join(save_dir, f'depths_sublist_{idx}.pkl'), 'rb') as f:
            all_depths.extend(pickle.load(f))
        with open(os.path.join(save_dir, f'depths_next_sublist_{idx}.pkl'), 'rb') as f:
            all_depths_next.extend(pickle.load(f))
        with open(os.path.join(save_dir, f'terrains_sublist_{idx}.pkl'), 'rb') as f:
            all_terrains.extend(pickle.load(f))

    with open(os.path.join(output_dir, 'small_train_val_depths.pkl'), 'wb') as f:
        pickle.dump(all_depths, f)
    with open(os.path.join(output_dir, 'small_train_val_depths_next.pkl'), 'wb') as f:
        pickle.dump(all_depths_next, f)
    with open(os.path.join(output_dir, 'small_train_val_terrains.pkl'), 'wb') as f:
        pickle.dump(all_terrains, f)

    # Assertions
    assert len(all_depths) == len(all_depths_next) == len(all_terrains), "Mismatch in number of samples."
    assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths), "Invalid shape in depths."
    assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths_next), "Invalid shape in depths_next."
    assert all(isinstance(item, np.ndarray) and item.shape == (321, 321) for item in all_terrains), "Invalid shape in terrains."

    logging.info("Small train/val subset saved and verified.")

# Run preprocessing for all projects
for prj_num, (prj_name, sublists) in prj_sublists.items():
    process_project(prj_num, prj_name, sublists)

# Concatenate train/val patches from all projects into unified lists
concatenate_all_train_val_chunks(prj_sublists)

# Concatenate test patches from all projects into unified lists
concatenate_all_test_chunks(prj_sublists)

# Concatenate prj_03 split separately
concatenate_prj03_split()

# Generate small_train_val set from prj_03
generate_small_train_val_subset()

