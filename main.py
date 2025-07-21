# main.py – Deep Learning Flood Modeling Pipeline

import argparse
import logging
import sys

from config.data_config import prjs_metadata

from simulations_to_samples.scripts.generate_patches import process_project
from simulations_to_samples.scripts.generate_datasets import create_and_save_datasets
from simulations_to_samples.scripts.generate_dataloaders import create_and_save_dataloaders

from multi_architecture_training.A_train_all_archs_on_small_set.train import run_train_all_on_small
from multi_architecture_training.A_train_all_archs_on_small_set.plot_all_losses import plot_all_losses as A_plot_all_losses
from multi_architecture_training.B_tune_one_arch_on_small_set.train import run_train_arch_04_tuned
from multi_architecture_training.B_tune_one_arch_on_small_set.plot_losses import plot_losses as B_plot_arch_04_losses   
from multi_architecture_training.C_train_best_four_on_big_set.train import run_train_best_four_on_big
from multi_architecture_training.C_train_best_four_on_big_set.plot_all_losses import plot_all_losses as C_plot_all_losses
from multi_architecture_training.training_utils.dummy_losses import calculate_and_save_dummy_losses

from evaluate_and_visualize_best_model.generate_rae_boxplot import plot_rae_boxplots_arch_05
from evaluate_and_visualize_best_model.visualize_predictions import plot_entire_batch


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
# Stage 2: Multi Architecture Training
# ----------------------------

def calculate_dummy_losses():
    logger.info("Calculating dummy losses for big_vald and small_val...")
    # Removed calculate_and_save_dummy_losses()
    pass # No longer needed
    
def A_train():
    logger.info("Training all architectures on small_train loader...")
    run_train_all_on_small()

def A_plot_losses():
    logger.info("Plotting all architectures' loss curves...")
    A_plot_all_losses()

def B_train():
    logger.info("Training Tuned Non-downsampling Convolutions with Attention on small set...")
    run_train_arch_04_tuned()

def B_plot_losses():
    logger.info("Plotting tuned Arch_04 loss curve...")
    B_plot_arch_04_losses()

def C_train():
    logger.info("Training best four architectures on big_train loader...")
    run_train_best_four_on_big()

def C_plot_losses():
    logger.info("Plotting best four losses...")
    C_plot_all_losses()

# ----------------------------
# Stage 3: Evaluation and Visualization
# ----------------------------
def plot_rae_boxplots():
    logger.info("Calculating and plotting RAE boxplots for ClassicUNet...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_rae_boxplots_arch_05(device)

def plot_entire_batch_predictions():
    plot_entire_batch() 
# ----------------------------
# CLI Argument Parser
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Flood Modeling Pipeline")
    
    parser.add_argument('task', type=str, help='Task to run', choices=[
        'generate_patches',
        'generate_datasets',
        'generate_dataloaders',
        'calculate_dummy_losses',
        'A_train',
        'A_plot_losses',
        'B_train',
        'B_plot_losses',
        'C_train',
        'C_plot_losses',
        'plot_rae_boxplots',
        'plot_entire_batch_predictions'
    ])
    
    args = parser.parse_args()

    task_map = {
        'generate_patches': generate_patches,
        'generate_datasets': generate_datasets,
        'generate_dataloaders': generate_dataloaders,       
        'calculate_dummy_losses': calculate_dummy_losses,
        'A_train': A_train,
        'A_plot_losses': A_plot_all_losses, 
        'B_train': B_train,
        'B_plot_losses': B_plot_losses,
        'C_train': C_train,
        'C_plot_losses': C_plot_losses,
        'plot_rae_boxplots': plot_rae_boxplots,
        'plot_entire_batch_predictions': plot_entire_batch_predictions
    }

    task_map[args.task]()

if __name__ == "__main__":
    main()
