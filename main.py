# main.py – Step 1 Only: Simulations → Datasets

import argparse
import logging
import sys

from config import prjs_metadata
from simulations_to_samples.scripts.generate_patches import process_project
from simulations_to_samples.scripts.generate_datasets import create_and_save_datasets
from simulations_to_samples.scripts.generate_dataloaders import create_and_save_dataloaders

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# ----------------------------
# Step 1: Simulations → Samples
# ----------------------------

def generate_patches():
    logger.info("Generating patches from HEC-RAS simulations...")
    for prj_num, (prj_name, plans) in prjs_metadata.items():
        process_project(prj_num, prj_name, plans)

def generate_datasets():
    logger.info("Creating dataset files...")
    create_and_save_datasets()

def generate_dataloaders():
    logger.info("Creating dataloaders...")
    create_and_save_dataloaders()

# ----------------------------
# CLI Argument Parser
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 1: Generate data from HEC-RAS simulations")
    
    parser.add_argument('task', type=str, help='Task to run', choices=[
        'generate_patches',
        'generate_datasets',
        'generate_dataloaders'
    ])
    
    args = parser.parse_args()

    task_map = {
        'generate_patches': generate_patches,
        'generate_datasets': generate_datasets,
        'generate_dataloaders': generate_dataloaders,
    }

    logger.info(f"Starting task: {args.task}")
    task_map[args.task]()

if __name__ == '__main__':
    main()
