# main.py – Deep Learning Flood Modeling Pipeline
import argparse
import logging
import sys

from config.data_config import prjs_metadata

from simulations_to_samples.scripts.generate_patches import process_project
from simulations_to_samples.scripts.generate_datasets import create_and_save_datasets
from simulations_to_samples.scripts.generate_dataloaders import create_and_save_dataloaders

from multi_architecture_training.B_train_all_archs_on_small_set.train import run_train_all_on_small
from multi_architecture_training.B_train_all_archs_on_small_set.plot_all_losses import plot_all_losses as B_plot_all_losses
from multi_architecture_training.A_tune_one_arch_on_small_set.train import run_train_arch_04_tuned
from multi_architecture_training.A_tune_one_arch_on_small_set.plot_losses import plot_losses as A_plot_arch_04_losses   
from multi_architecture_training.C_train_best_four_on_big_set.train import run_train_best_four_on_big
from multi_architecture_training.C_train_best_four_on_big_set.plot_all_losses import plot_all_losses as C_plot_all_losses
from multi_architecture_training.training_utils.dummy_losses import calculate_and_save_dummy_losses

from evaluate_and_visualize_best_model.scripts.generate_rae_boxplot import plot_rae_boxplots_arch_04
from evaluate_and_visualize_best_model.scripts.visualize_predictions import plot_entire_batch

from full_domain_closure_best_model.apply_closure_on_simulations import run_closure_on_all_simulations

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
    logger.info("Calculating dummy losses for big_val and small_val...")
    calculate_and_save_dummy_losses()
    
def A_train():
    logger.info("Training Tuned Non-downsampling Convolutions with Attention on small set...")
    run_train_arch_04_tuned()

def A_plot_losses():
    logger.info("Plotting tuned Arch_04 loss curve...")
    A_plot_arch_04_losses()

def B_train():
    logger.info("Training all architectures on small_train loader...")
    run_train_all_on_small()

def B_plot_losses():
    logger.info("Plotting all architectures' loss curves...")
    B_plot_all_losses()

def C_train():
    logger.info("Training best four architectures on big_train loader...")
    run_train_best_four_on_big()

def C_plot_losses():
    logger.info("Plotting best four losses...")
    C_plot_all_losses()

# ----------------------------
# Stage 3: Evaluation and Visualization
# ----------------------------
def generate_rae_boxplots():
    logger.info("Calculating and plotting RAE boxplots for Non-downsampling Convolutions with Self-Attention...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_rae_boxplots_arch_04(device)

def plot_entire_batch_predictions():
    logger.info("Plotting entire batch predictions...")
    plot_entire_batch() 
# ----------------------------
# Stage 4: Full Domain Closure
# ----------------------------
def run_closure():
    logger.info("Running full domain closure on all simulations...")
    run_closure_on_all_simulations()

# ----------------------------
# Complete Pipeline
# ----------------------------
def run_all():
    """Run the complete pipeline from start to finish"""
    logger.info("Starting complete Deep Learning Flood Modeling Pipeline...")
    
    # Stage 1: Simulations → Samples
    logger.info("=" * 60)
    logger.info("STAGE 1: SIMULATIONS → SAMPLES")
    logger.info("=" * 60)
    generate_patches()
    generate_datasets()
    generate_dataloaders()
    
    # Stage 2: Multi Architecture Training
    logger.info("=" * 60)
    logger.info("STAGE 2: MULTI ARCHITECTURE TRAINING")
    logger.info("=" * 60)
    calculate_dummy_losses()
    A_train()
    A_plot_losses()
    B_train()
    B_plot_losses()
    C_train()
    C_plot_losses()
    
    # Stage 3: Evaluation and Visualization
    logger.info("=" * 60)
    logger.info("STAGE 3: EVALUATION AND VISUALIZATION")
    logger.info("=" * 60)
    generate_rae_boxplots()
    plot_entire_batch_predictions()
    
    """
    # Stage 4: Full Domain Closure
    logger.info("=" * 60)
    logger.info("STAGE 4: FULL DOMAIN CLOSURE")
    logger.info("=" * 60)
    run_closure()
    """
    
    logger.info("=" * 60)
    logger.info("Complete pipeline finished successfully!")
    logger.info("=" * 60)

# ----------------------------
# CLI Argument Parser
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Flood Modeling Pipeline")
    
    parser.add_argument('task', type=str, help='Task to run', choices=[
        'run_all',
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
        'generate_rae_boxplots',
        'plot_entire_batch_predictions',
        'run_closure'
    ])
    
    args = parser.parse_args()

    task_map = {
        'run_all': run_all,
        'generate_patches': generate_patches,
        'generate_datasets': generate_datasets,
        'generate_dataloaders': generate_dataloaders,       
        'calculate_dummy_losses': calculate_dummy_losses,
        'A_train': A_train,
        'A_plot_losses': A_plot_losses, 
        'B_train': B_train,
        'B_plot_losses': B_plot_losses,
        'C_train': C_train,
        'C_plot_losses': C_plot_losses,
        'generate_rae_boxplots': generate_rae_boxplots,
        'plot_entire_batch_predictions': plot_entire_batch_predictions,
        'run_closure': run_closure
    }

    task_map[args.task]()

if __name__ == "__main__":
    main()
