# Simulations to Samples Pipeline

This directory contains the data processing pipeline that converts HEC-RAS simulation outputs into deep learning training data.

## 📁 Directory Structure

```
simulations_to_samples/
├── raw_data/                          # Raw HEC-RAS simulation outputs
│   ├── hecras_simulations_results/    # HEC-RAS project results
│   │   ├── prj_03/                    # Project 03 simulation files
│   │   ├── prj_04/                    # Project 04 simulation files
│   │   ├── prj_05/                    # Project 05 simulation files
│   │   └── prj_06/                    # Project 06 simulation files
│   └── images/                        # Visualization images
├── processed_data/                    # Processed data outputs
│   ├── patches_per_simulation/        # Extracted patches per simulation
│   ├── datasets/                      # Combined dataset files
│   │   ├── small_dataset/             # Small training dataset
│   │   ├── big_dataset/               # Large training dataset
│   │   └── prj_03_dataset/            # Project 03 specific dataset
│   └── dataloaders/                   # PyTorch DataLoaders
│       └── figures/                   # Sample visualizations
└── scripts/                           # Processing scripts
    ├── generate_patches.py            # Extract patches from simulations
    ├── generate_datasets.py           # Create dataset files
    ├── generate_dataloaders.py        # Create PyTorch DataLoaders
    └── patch_extractor_processor.py   # Core patch extraction logic
```

## 🔄 Processing Pipeline

### Stage 1: Patch Extraction
- **Input**: HEC-RAS simulation results
- **Process**: Extract 32x32 patches from simulation outputs
- **Output**: Individual patch files per simulation
- **Script**: `generate_patches.py`

### Stage 2: Dataset Creation
- **Input**: Individual patch files
- **Process**: Combine patches into training/validation/test datasets
- **Output**: Dataset pickle files
- **Script**: `generate_datasets.py`

### Stage 3: DataLoader Generation
- **Input**: Dataset files
- **Process**: Create PyTorch DataLoaders with proper formatting
- **Output**: PyTorch DataLoader files (.pt)
- **Script**: `generate_dataloaders.py`

## 📊 Dataset Types

### Small Dataset
- **Purpose**: Quick testing and development
- **Size**: ~7 simulations from prj_03
- **Files**: `small_train_loader.pt`, `small_val_loader.pt`

### Big Dataset
- **Purpose**: Full model training
- **Size**: All simulations except test sets
- **Files**: `big_train_loader.pt`, `big_val_loader.pt`, `big_test_loader.pt`

### Project 03 Dataset
- **Purpose**: Project-specific analysis
- **Size**: All prj_03 simulations
- **Files**: `prj_03_train_val_loader.pt`, `prj_03_test_loader.pt`

## 🎯 Data Format

### Input Data (Per Patch)
- **Terrain**: 32x32 elevation data
- **Depth**: 32x32 water depth at time n
- **Depth Next**: 32x32 water depth at time n+1

### Output Data (Per Patch)
- **Input**: 3-channel tensor (terrain, depth, boundary conditions)
- **Target**: 1-channel tensor (depth difference: depth_next - depth)

## 🚀 Usage

### Generate All Data
```bash
python main.py generate_patches
python main.py generate_datasets
python main.py generate_dataloaders
```

### Individual Steps
```bash
# Extract patches from simulations
python -c "from simulations_to_samples.scripts.generate_patches import process_project; from config.data_config import prjs_metadata; [process_project(prj_num, prj_name, plans) for prj_num, (prj_name, plans) in prjs_metadata.items()]"

# Create datasets
python -c "from simulations_to_samples.scripts.generate_datasets import create_and_save_datasets; create_and_save_datasets()"

# Create dataloaders
python -c "from simulations_to_samples.scripts.generate_dataloaders import create_and_save_dataloaders; create_and_save_dataloaders()"
```

## ⚙️ Configuration

Key parameters are defined in `config/data_config.py`:
- `PATCH_SIZE`: Size of extracted patches (32)
- `BOUNDARY_THICKNESS`: Boundary thickness for BC (2)
- `BATCH_SIZE`: Training batch size (300)
- `prjs_metadata`: Project and plan configurations

## 📈 File Sizes

- **Patches**: ~16MB per simulation
- **Small Dataset**: ~483MB per loader
- **Big Dataset**: ~13GB per loader
- **Total Processed Data**: ~50GB

## 🔧 Troubleshooting

### Common Issues
1. **Memory Errors**: Large datasets may require more RAM
2. **File Not Found**: Ensure HEC-RAS simulation files exist in `raw_data/`
3. **Import Errors**: Check that config files are properly set up

### Validation
- Check generated figures in `dataloaders/figures/` for data quality
- Verify file sizes match expected values
- Test DataLoader loading in training scripts 