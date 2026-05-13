# multi_architecture_training/A_tune_one_arch_on_small_set/arch_04_tuned_config.py
# Architecture configuration for Stage A - using centralized config system

from config.model_configs import get_stage_A_config

# Get configuration from centralized config system
final_config = get_stage_A_config()
