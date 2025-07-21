# config/training_config.py
# Training hyperparameters and configurations

# Default training parameters
DEFAULT_EPOCHS = 300
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_BATCH_SIZE = 300

# Weight initialization
DEFAULT_WEIGHT_INIT = 'xavier'

# Optimizer settings
OPTIMIZER_TYPE = 'Adam'
OPTIMIZER_BETA1 = 0.5
OPTIMIZER_BETA2 = 0.999

# Loss function
LOSS_FUNCTION = 'L1Loss'

# Device configuration
DEVICE = 'cuda'  # Will fall back to 'cpu' if CUDA not available

# Training stages
TRAINING_STAGES = {
    'A': 'train_all_archs_on_small_set',
    'B': 'tune_one_arch_on_small_set', 
    'C': 'train_best_four_on_big_set'
}

# Model saving
SAVE_MODELS = True
SAVE_LOSSES = True
SAVE_PLOTS = True

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = "%(asctime)s — %(levelname)s — %(message)s" 