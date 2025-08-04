# Multi-Architecture Training Pipeline

This directory contains the complete training pipeline for multiple deep learning architectures for flood modeling.

## 📁 Directory Structure

```
multi_architecture_training/
├── README.md                           # This documentation
├── models/                             # All model architectures
│   ├── classic_unet.py                 # Classic UNet implementation
│   ├── simplified_unet.py              # Simplified UNet
│   ├── non_downsampling_convolutions.py # Basic convolutions
│   ├── non_downsampling_convolutions_attention.py # Convolutions with attention
│   ├── encoder_decoder_attention.py    # Encoder-decoder with attention
│   ├── unet_resnet_modified.py         # UNet with ResNet modifications
│   ├── encoder_decoder_large_convolutions.py # Large convolution encoder-decoder
│   ├── terrain_downsampler_*.py        # Various terrain downsampling methods
│   └── attention.py                    # Attention mechanism implementations
├── training_utils/                     # Shared training utilities
│   ├── train_model.py                  # Core training function
│   ├── training_monitor.py             # Training monitoring and early stopping
│   ├── weights_init.py                 # Weight initialization
│   ├── dummy_loss.py                   # Dummy loss calculation
│   └── dummy_val_loss.pt               # Dummy validation loss file
├── A_train_all_archs_on_small_set/     # Stage A: Train all architectures on small dataset
│   ├── train.py                        # Training script
│   ├── plot_all_losses.py              # Loss plotting
│   ├── architecture_configs.py         # Architecture configurations
│   ├── saved_trained_models/           # Trained models for this stage
│   ├── saved_losses/                   # Training/validation losses for this stage
│   └── learning_curves.png             # Generated learning curves
├── B_tune_one_arch_on_small_set/       # Stage B: Tune Arch_04 on small dataset
│   ├── train.py                        # Training script
│   ├── plot_losses.py                  # Loss plotting
│   ├── arch_04_tuned_config.py         # Tuned configuration
│   ├── saved_trained_models/           # Trained models for this stage
│   ├── saved_losses/                   # Training/validation losses for this stage
│   └── tuned_arch_04_loss_plot.png     # Generated loss plot
├── C_train_best_four_on_big_set/       # Stage C: Train best four on big dataset
│   ├── train.py                        # Training script
│   ├── plot_all_losses.py              # Loss plotting
│   ├── architecture_configs.py         # Best four configurations
│   ├── saved_trained_models/           # Trained models for this stage
│   └── saved_losses/                   # Training/validation losses for this stage
└── D_evaluation/                       # Stage D: Evaluation and analysis
    ├── boxplots_RAE_all_sets/          # RAE analysis and boxplots
    ├── visualize_prediction_and_errors/ # Prediction visualization
    └── compare_all_archs_runtime_size_performance/ # Architecture comparison
```

## 🔄 Training Pipeline Stages

### **Stage A: Train All Architectures on Small Dataset**
- **Purpose**: Evaluate all architectures on small dataset
- **Architectures**: Arch_02, Arch_03, Arch_04, Arch_05, Arch_07, Arch_08, Arch_09
- **Dataset**: Small dataset (~7 simulations)
- **Output**: Performance comparison and loss curves
- **Results Location**: `A_train_all_archs_on_small_set/saved_trained_models/` and `saved_losses/`

### **Stage B: Tune One Architecture on Small Dataset**
- **Purpose**: Fine-tune Arch_04 (best performing) on small dataset
- **Architecture**: Arch_04 (Non-downsampling Convolutions with Attention)
- **Dataset**: Small dataset
- **Output**: Optimized Arch_04 configuration
- **Results Location**: `B_tune_one_arch_on_small_set/saved_trained_models/` and `saved_losses/`

### **Stage C: Train Best Four on Big Dataset**
- **Purpose**: Train top-performing architectures on full dataset
- **Architectures**: Arch_03, Arch_04 (tuned), Arch_05, Arch_07
- **Dataset**: Big dataset (all simulations except test)
- **Output**: Production-ready models
- **Results Location**: `C_train_best_four_on_big_set/saved_trained_models/` and `saved_losses/`

### **Stage D: Evaluation and Analysis**
- **Purpose**: Comprehensive evaluation and comparison
- **Activities**: RAE analysis, visualization, performance comparison
- **Output**: Final results and insights
- **Results Location**: `D_evaluation/` subdirectories

## 🏗️ Model Architectures

### **Core Architectures**
1. **Classic UNet** (`classic_unet.py`): Standard UNet implementation
2. **Simplified UNet** (`simplified_unet.py`): Lightweight UNet variant
3. **Non-downsampling Convolutions** (`non_downsampling_convolutions.py`): Basic convolutional network
4. **Non-downsampling Convolutions with Attention** (`non_downsampling_convolutions_attention.py`): Convolutions with self-attention
5. **Encoder-Decoder with Attention** (`encoder_decoder_attention.py`): Attention-based encoder-decoder
6. **UNet ResNet Modified** (`unet_resnet_modified.py`): UNet with ResNet components
7. **Encoder-Decoder Large Convolutions** (`encoder_decoder_large_convolutions.py`): Large kernel encoder-decoder

### **Terrain Downsamplers**
- **Alternating** (`terrain_downsampler_alternating.py`): Alternating stride downsampling
- **K11S10** (`terrain_downsampler_k11s10.py`): 11x11 kernel, stride 10
- **Cubic** (`terrain_downsampler_cubic.py`): Cubic interpolation downsampling

### **Attention Mechanisms**
- **Self-Attention** (`self_attention.py`): Standard self-attention
- **Attention** (`attention.py`): General attention implementation

## 🚀 Usage

### **Run Complete Pipeline**
```bash
# Stage A: Train all architectures
python main.py A_train
python main.py A_plot_losses

# Stage B: Tune Arch_04
python main.py B_train
python main.py B_plot_losses

# Stage C: Train best four
python main.py C_train
python main.py C_plot_losses
```

### **Individual Stage Execution**
```bash
# Train specific stage
python -c "from multi_architecture_training.A_train_all_archs_on_small_set.train import run_train_all_on_small; run_train_all_on_small()"

# Plot losses
python -c "from multi_architecture_training.A_train_all_archs_on_small_set.plot_all_losses import plot_all_losses; plot_all_losses()"
```

## ⚙️ Configuration

### **Architecture Configurations**
- **Stage A**: `A_train_all_archs_on_small_set/architecture_configs.py`
- **Stage B**: `B_tune_one_arch_on_small_set/arch_04_tuned_config.py`
- **Stage C**: `C_train_best_four_on_big_set/architecture_configs.py`

### **Training Parameters**
- **Loss Function**: L1Loss
- **Optimizer**: Adam
- **Learning Rate**: 0.0001 (configurable per architecture)
- **Epochs**: 300 (configurable per architecture)
- **Batch Size**: 300 (from data config)

## 📊 Results and Outputs

### **Decentralized Results Storage**
Each training stage saves its own results locally:

#### **Stage A Results**
- **Models**: `A_train_all_archs_on_small_set/saved_trained_models/{arch_name}/model.pth`
- **Losses**: `A_train_all_archs_on_small_set/saved_losses/{arch_name}/losses.pt`
- **Plots**: `A_train_all_archs_on_small_set/learning_curves.png`

#### **Stage B Results**
- **Models**: `B_tune_one_arch_on_small_set/saved_trained_models/Arch_04/model.pth`
- **Losses**: `B_tune_one_arch_on_small_set/saved_losses/Arch_04/losses.pt`
- **Plots**: `B_tune_one_arch_on_small_set/tuned_arch_04_loss_plot.png`

#### **Stage C Results**
- **Models**: `C_train_best_four_on_big_set/saved_trained_models/{arch_name}/model.pth`
- **Losses**: `C_train_best_four_on_big_set/saved_losses/{arch_name}/losses.pt`

### **File Formats**
- **Models**: PyTorch state dict (.pth)
- **Losses**: PyTorch tensor with train/val losses (.pt)
- **Plots**: PNG images

## 🔧 Training Utilities

### **Core Functions**
- `train_model()`: Main training loop with progress tracking
- `TrainingMonitor`: Advanced training monitoring with early stopping
- `weights_init()`: Xavier weight initialization
- `calculate_dummy_mean_loss()`: Dummy loss calculation for comparison

### **Features**
- Progress bars with real-time loss display
- Automatic model and loss saving per stage
- Device detection (CUDA/CPU)
- Comprehensive logging
- Early stopping to prevent overfitting
- Convergence analysis and overfitting detection

## 📈 Performance Monitoring

### **Metrics Tracked**
- Training loss per epoch
- Validation loss per epoch
- Dummy validation loss (baseline comparison)
- Model convergence patterns
- Early stopping triggers

### **Visualization**
- Learning curves for all architectures
- Loss comparison plots
- Training progress tracking
- Convergence analysis plots

## 🛠️ Troubleshooting

### **Common Issues**
1. **CUDA Out of Memory**: Reduce batch size or use smaller models
2. **Training Not Converging**: Check learning rate and model configuration
3. **File Not Found**: Ensure data pipeline has been run first

### **Best Practices**
1. **Monitor Losses**: Check for overfitting/underfitting
2. **Save Checkpoints**: Models are automatically saved per stage
3. **Validate Data**: Use data validation before training
4. **Check Resources**: Monitor GPU memory and disk space

## 🔄 Integration with Main Pipeline

This directory integrates with the main project through:
- **Data**: Uses processed dataloaders from `simulations_to_samples/`
- **Config**: Uses centralized configuration from `config/`
- **Main Entry**: Controlled through `main.py` commands
- **Results**: Decentralized storage per training stage

## 📁 Results Organization Benefits

### **Advantages of Decentralized Storage**
1. **Stage Isolation**: Each stage's results are self-contained
2. **Easy Cleanup**: Can remove individual stage results without affecting others
3. **Clear Organization**: Results are co-located with their training scripts
4. **Independent Execution**: Each stage can be run independently
5. **Version Control**: Easier to track changes per stage
6. **Backup Management**: Can backup/restore individual stages 