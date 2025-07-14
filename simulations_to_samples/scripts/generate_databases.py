
# This script processes HEC-RAS simulation outputs into patch data for deep learning.
# Each number in the sublists corresponds to a plan ID (plan_num) — a simulation run in HEC-RAS.
# For memory efficiency, we process simulations in groups (sublists) using PatchExtractorProcessor.
# Output patches are saved per group under ./Database/saved_in_chunks/{prj_num}/.
# Then all training/validation chunks are concatenated across all projects (excluding test groups).

import os
import pickle
import numpy as np
import logging
from simulations_to_samples.scripts.patch_extractor_processor import PatchExtractorProcessor
from config import prj_sublists, PATCHES_ROOT, DATABASES_ROOT, DATALOADERS_ROOT

logging.basicConfig(level=logging.INFO)

        
def concatenate_all_train_val_chunks(prj_sublists):
    all_depths, all_depths_next, all_terrains = [], [], []

    for prj_num, (_, sublists) in prj_sublists.items():
        save_dir = os.path.join(DATA_ROOT, f'saved_in_chunks/{prj_num}/')
        last_index = len(sublists) - 1

        for idx in range(last_index):
            all_depths.extend(load_pickle(os.path.join(save_dir, f'depths_sublist_{idx}.pkl')))
            all_depths_next.extend(load_pickle(os.path.join(save_dir, f'depths_next_sublist_{idx}.pkl')))
            all_terrains.extend(load_pickle(os.path.join(save_dir, f'terrains_sublist_{idx}.pkl')))

    with open(os.path.join(DATA_ROOT, 'big_train_val_depths.pkl'), 'wb') as f:
        pickle.dump(all_depths, f)
    with open(os.path.join(DATA_ROOT, 'big_train_val_depths_next.pkl'), 'wb') as f:
        pickle.dump(all_depths_next, f)
    with open(os.path.join(DATA_ROOT, 'big_train_val_terrains.pkl'), 'wb') as f:
        pickle.dump(all_terrains, f)

    logging.info("Final big_train_val files saved across all projects.")

    assert len(all_depths) == len(all_depths_next) == len(all_terrains), "Mismatch in number of samples across lists."
    assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths)
    assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths_next)
    assert all(isinstance(item, np.ndarray) and item.shape == (321, 321) for item in all_terrains)
    logging.info("All checks passed successfully for final combined lists.")

def concatenate_all_test_chunks(prj_sublists):
    all_depths, all_depths_next, all_terrains = [], [], []

    for prj_num, (_, sublists) in prj_sublists.items():
        save_dir = os.path.join(DATA_ROOT, f'saved_in_chunks/{prj_num}/')
        last_index = len(sublists) - 1

        all_depths.extend(load_pickle(os.path.join(save_dir, f'depths_sublist_{last_index}.pkl')))
        all_depths_next.extend(load_pickle(os.path.join(save_dir, f'depths_next_sublist_{last_index}.pkl')))
        all_terrains.extend(load_pickle(os.path.join(save_dir, f'terrains_sublist_{last_index}.pkl')))

    with open(os.path.join(DATA_ROOT, 'big_test_depths.pkl'), 'wb') as f:
        pickle.dump(all_depths, f)
    with open(os.path.join(DATA_ROOT, 'big_test_depths_next.pkl'), 'wb') as f:
        pickle.dump(all_depths_next, f)
    with open(os.path.join(DATA_ROOT, 'big_test_terrains.pkl'), 'wb') as f:
        pickle.dump(all_terrains, f)

    logging.info("Final big_test files saved across all projects.")

def concatenate_prj03_train_val_test_chunks():
    prj_num = '03'
    save_dir = os.path.join(DATA_ROOT, f'saved_in_chunks/{prj_num}/')
    sublists = prj_sublists[prj_num][1]

    train_val_depths, train_val_depths_next, train_val_terrains = [], [], []
    test_depths, test_depths_next, test_terrains = [], [], []

    for idx in range(len(sublists)):
        depths = load_pickle(os.path.join(save_dir, f'depths_sublist_{idx}.pkl'))
        depths_next = load_pickle(os.path.join(save_dir, f'depths_next_sublist_{idx}.pkl'))
        terrains = load_pickle(os.path.join(save_dir, f'terrains_sublist_{idx}.pkl'))

        if idx == len(sublists) - 1:
            test_depths.extend(depths)
            test_depths_next.extend(depths_next)
            test_terrains.extend(terrains)
        else:
            train_val_depths.extend(depths)
            train_val_depths_next.extend(depths_next)
            train_val_terrains.extend(terrains)

    with open(os.path.join(DATA_ROOT, 'prj_03_train_val_depths.pkl'), 'wb') as f:
        pickle.dump(train_val_depths, f)
    with open(os.path.join(DATA_ROOT, 'prj_03_train_val_depths_next.pkl'), 'wb') as f:
        pickle.dump(train_val_depths_next, f)
    with open(os.path.join(DATA_ROOT, 'prj_03_train_val_terrains.pkl'), 'wb') as f:
        pickle.dump(train_val_terrains, f)

    with open(os.path.join(DATA_ROOT, 'prj_03_test_depths.pkl'), 'wb') as f:
        pickle.dump(test_depths, f)
    with open(os.path.join(DATA_ROOT, 'prj_03_test_depths_next.pkl'), 'wb') as f:
        pickle.dump(test_depths_next, f)
    with open(os.path.join(DATA_ROOT, 'prj_03_test_terrains.pkl'), 'wb') as f:
        pickle.dump(test_terrains, f)

    logging.info("prj_03 train/val and test files saved.")

def generate_small_train_val_subset():
    prj_num = '03'
    sublists = prj_sublists[prj_num][1]
    save_dir = os.path.join(DATA_ROOT, f'saved_in_chunks/{prj_num}/')

    all_depths, all_depths_next, all_terrains = [], [], []

    for idx in range(3):
        all_depths.extend(load_pickle(os.path.join(save_dir, f'depths_sublist_{idx}.pkl')))
        all_depths_next.extend(load_pickle(os.path.join(save_dir, f'depths_next_sublist_{idx}.pkl')))
        all_terrains.extend(load_pickle(os.path.join(save_dir, f'terrains_sublist_{idx}.pkl')))

    with open(os.path.join(DATA_ROOT, 'small_train_val_depths.pkl'), 'wb') as f:
        pickle.dump(all_depths, f)
    with open(os.path.join(DATA_ROOT, 'small_train_val_depths_next.pkl'), 'wb') as f:
        pickle.dump(all_depths_next, f)
    with open(os.path.join(DATA_ROOT, 'small_train_val_terrains.pkl'), 'wb') as f:
        pickle.dump(all_terrains, f)

    assert len(all_depths) == len(all_depths_next) == len(all_terrains), "Mismatch in number of samples."
    assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths)
    assert all(isinstance(item, np.ndarray) and item.shape == (32, 32) for item in all_depths_next)
    assert all(isinstance(item, np.ndarray) and item.shape == (321, 321) for item in all_terrains)

    logging.info("Small train/val subset saved and verified.")
