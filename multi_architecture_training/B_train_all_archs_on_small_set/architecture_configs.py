# multi_architecture_training/B_train_all_archs_on_small_set/architecture_configs.py
# Architecture configurations for Stage B - using centralized config system

from config.model_configs import get_stage_B_configs

# Get configurations from centralized config system
architectures = get_stage_B_configs()
