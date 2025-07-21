# multi_architecture_training/A_train_all_archs_on_small_set/architecture_configs.py
# Architecture configurations for Stage A - using centralized config system

from config.model_configs import get_stage_A_configs

# Get configurations from centralized config system
architectures = get_stage_A_configs()
