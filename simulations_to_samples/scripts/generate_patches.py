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
from config import PATCHES_ROOT

# Create the root directory and 'pkls' subdirectory
PKLS_DIR = os.path.join(PATCHES_ROOT, 'patches_per_simulation_pkls')
os.makedirs(PKLS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

def process_project(prj_num, prj_name, plans):
    for plan_num in plans:
        try: 
            instance = PatchExtractorProcessor(prj_num, prj_name, plan_num)
            instance.generate_patches()  

            # Save patch data inside the 'pkls' directory
            with open(os.path.join(PKLS_DIR, f'prj_{prj_num}_plan_{plan_num}_terrain_patches.pkl'), 'wb') as f:
                pickle.dump(instance.database['terrain'], f)
            with open(os.path.join(PKLS_DIR, f'prj_{prj_num}_plan_{plan_num}_depth_patches.pkl'), 'wb') as f:
                pickle.dump(instance.database['depth'], f)
            with open(os.path.join(PKLS_DIR, f'prj_{prj_num}_plan_{plan_num}_depth_next_patches.pkl'), 'wb') as f:
                pickle.dump(instance.database['depth_next'], f)
                
        except Exception as e:
            logging.error(f"Failed to process prj_{prj_num}, plan_{plan_num}: {e}")

