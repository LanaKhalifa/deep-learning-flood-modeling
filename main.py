# main.py – Command-line interface for full pipeline

import argparse
import logging
import sys
from config import prjs_metadata
from simulations_to_samples.scripts.generate_patches import process_project
from simulations_to_samples.scripts.generate_datasets import create_and_save_datasets
from simulations_to_samples.scripts.generate_dataloaders import create_and_save_dataloaders

from multi_architecture_training.A_train_all_archs_on_small_set.train import run_train_all_on_small
from multi_architecture_training.A_train_all_archs_on_small_set.plot_all_losses import plot_all_losses
from multi_architecture_training.B_tune_one_arch_on_small_set.train import run_train_final_model
from multi_architecture_training.B_tune_one_arch_on_small_set.plot_loss import plot_final_model_loss


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
# Stage 1: Simulations → Samples
# ----------------------------

def generate_patches():
    logger.info("🔷 Generating patches from HEC-RAS simulations...")
    for prj_num, (prj_name, plans) in prjs_metadata.items():
        process_project(prj_num, prj_name, plans)

def generate_datasets():
    logger.info("🔷 Creating dataset files...")
    create_and_save_datasets()

def generate_dataloaders():
    logger.info("🔷 Creating dataloaders...")
    create_and_save_dataloaders()


# ----------------------------
# Stage 2: Training
# ----------------------------

def train_all():
    logger.info("🔷 Training all architectures on small set...")
    run_train_all_on_small()

def plot_all():
    logger.info("📊 Plotting all architectures' loss curves...")
    plot_all_losses()

def tune_arch_04():
    logger.info("🔷 Training final tuned Arch_04 model...")
    run_train_final_model()

def plot_tuned():
    logger.info("📈 Plotting tuned Arch_04 loss curve...")
    plot_final_model_loss()


# ----------------------------
# CLI Argument Parser
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Flood Modeling Pipeline")
    
    parser.add_argument('task', type=str, help='Task to run', choices=[
        'generate_patches',
        'generate_datasets',
        'generate_dataloaders',
        'train_all',
        'plot_all',
        'tune_arch_04',
        'plot_tuned'
    ])
    
    args = parser.parse_args()

    task_map = {
        'generate_patches': generate_patches,
        'generate_datasets': generate_datasets,
        'generate_dataloaders': generate_dataloaders,
        'train_all': train_all,
        'plot_all': plot_all,
        'tune_arch_04': tune_arch_04,
        'plot_tuned': plot_tuned
    }

    logger.info(f"🚀 Starting task: {args.task}")
    task_map[args.task]()


if __name__ == '__main__':
    main()
