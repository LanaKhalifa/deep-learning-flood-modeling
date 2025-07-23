# config.py - Backward compatibility for existing imports
# This file imports from the new config system to maintain compatibility

from config.data_config import *
from config.paths_config import *
from config.training_config import *
from config.model_configs import *

# Legacy compatibility - map old names to new config system
DATALOADERS_DIR = str(DATALOADERS_DIR)
DIR_DATALOADERS = DATALOADERS_DIR
PATCHES_DIR = str(PATCHES_DIR)
DATASETS_DIR = str(DATASETS_DIR)

# Ensure all directories exist
ensure_directories()
