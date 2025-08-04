# config/paths_config.py
# File paths and directory configurations

import os
from pathlib import Path

# Base project directory
PROJECT_DIR = Path(__file__).parent.parent

# Data directories
SIMULATIONS_DIR = PROJECT_DIR / 'simulations_to_samples'
RAW_DATA_DIR = SIMULATIONS_DIR / 'raw_data'
PROCESSED_DATA_DIR = SIMULATIONS_DIR / 'processed_data'
PATCHES_DIR = PROCESSED_DATA_DIR / 'patches_per_simulation'
DATASETS_DIR = PROCESSED_DATA_DIR / 'datasets'
DATALOADERS_DIR = PROCESSED_DATA_DIR / 'dataloaders'
RAW_SIMULATIONS_DIR = RAW_DATA_DIR / 'hecras_simulations_results'

# Training directories
TRAINING_DIR = PROJECT_DIR / 'multi_architecture_training'
MODELS_DIR = TRAINING_DIR / 'models'
TRAINING_UTILS_DIR = TRAINING_DIR / 'training_utils'

# Stage-specific directories
STAGE_A_DIR = TRAINING_DIR / 'A_tune_one_arch_on_small_set'
STAGE_B_DIR = TRAINING_DIR / 'B_train_all_archs_on_small_set'
STAGE_C_DIR = TRAINING_DIR / 'C_train_best_four_on_big_set'

# Evaluation and Visualization directories
EVAL_VIS_DIR = PROJECT_DIR / 'evaluate_and_visualize_best_model'
RAE_BOXPLOTS_DIR = EVAL_VIS_DIR / 'boxplots'
VISUALIZATION_DIR = EVAL_VIS_DIR / 'visualizations'

# Stage-specific results directories
STAGE_A_MODELS_DIR = STAGE_A_DIR / 'saved_trained_models'
STAGE_A_LOSSES_DIR = STAGE_A_DIR / 'saved_losses'
STAGE_B_MODELS_DIR = STAGE_B_DIR / 'saved_trained_models'
STAGE_B_LOSSES_DIR = STAGE_B_DIR / 'saved_losses'
STAGE_C_MODELS_DIR = STAGE_C_DIR / 'saved_trained_models'
STAGE_C_LOSSES_DIR = STAGE_C_DIR / 'saved_losses'
BEST_MODEL_PATH = STAGE_C_MODELS_DIR / 'Arch_05' / 'model.pth'

# Full domain closure
FULL_DOMAIN_DIR = PROJECT_DIR / 'full_domain_closure_best_model'


def get_model_path(stage, arch_name):
    """Get path for a saved model based on stage"""
    if stage == 'A':
        return STAGE_A_MODELS_DIR / arch_name / 'model.pth'
    elif stage == 'B':
        return STAGE_B_MODELS_DIR / arch_name / 'model.pth'
    elif stage == 'C':
        return STAGE_C_MODELS_DIR / arch_name / 'model.pth'
    else:
        raise ValueError(f"Unknown stage: {stage}")

def get_losses_path(stage, arch_name):
    """Get path for saved losses based on stage"""
    if stage == 'A':
        return STAGE_A_LOSSES_DIR / arch_name / 'losses.pt'
    elif stage == 'B':
        return STAGE_B_LOSSES_DIR / arch_name / 'losses.pt'
    elif stage == 'C':
        return STAGE_C_LOSSES_DIR / arch_name / 'losses.pt'
    else:
        raise ValueError(f"Unknown stage: {stage}") 

# Create directories if they don't exist
def ensure_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PATCHES_DIR,
        DATASETS_DIR,
        DATALOADERS_DIR,
        MODELS_DIR,
        TRAINING_UTILS_DIR,
        STAGE_A_DIR,
        STAGE_B_DIR,
        STAGE_C_DIR,
        EVAL_VIS_DIR,
        RAE_BOXPLOTS_DIR,
        VISUALIZATION_DIR,
        STAGE_A_MODELS_DIR,
        STAGE_A_LOSSES_DIR,
        STAGE_B_MODELS_DIR,
        STAGE_B_LOSSES_DIR,
        STAGE_C_MODELS_DIR,
        STAGE_C_LOSSES_DIR,
        FULL_DOMAIN_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

# Data file paths
def get_dataloader_path(loader_name):
    """Get path for a specific dataloader"""
    return DATALOADERS_DIR / f'{loader_name}.pt'

def get_dataset_path(dataset_name):
    """Get path for a specific dataset"""
    return DATASETS_DIR / f'{dataset_name}.pkl'


