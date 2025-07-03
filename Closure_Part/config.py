
"""
config.py

Configuration constants used across modules.
"""

# Number of cells in each patch (used during training)
CELLS_IN_PATCH = 32

# Time difference between current and next step (in minutes)
DELTA_T = 60

# Cell resolution used in HEC-RAS (in meters)
METERS_IN_CELL = 10

# Model architecture and trial settings
ARCHITECTURE_NAME = "Arch_05"
TRIAL_NAME = "trial_final"

# Base paths
BASE_PROJECT_PATH = "/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final"
MODEL_DIR = f"{BASE_PROJECT_PATH}/{ARCHITECTURE_NAME}/{TRIAL_NAME}/trained_models"
MODEL_PATH = f"{MODEL_DIR}/model.pth"

# Plotting
FONT_NAME = "Nimbus Roman"
FONT_SIZE = 60
TICK_INTERVAL = 64
"""

Note: Make sure these paths are consistent with your actual project layout.
"""
