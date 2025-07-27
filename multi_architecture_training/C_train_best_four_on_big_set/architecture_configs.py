# multi_architecture_training/C_train_best_four_on_big_set/architecture_configs.py
# Architecture configurations for Stage C - using centralized config system

from config.model_configs import get_stage_C_configs

# Get configurations from centralized config system
architectures = get_stage_C_configs()
