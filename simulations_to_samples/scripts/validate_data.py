# simulations_to_samples/scripts/validate_data.py
"""
Validate the data pipeline and check data integrity.
Uses the new config system for paths.
"""

import os
import torch
import numpy as np
from pathlib import Path
from config.data_config import DATALOADERS_ROOT, DATASETS_ROOT, PATCHES_ROOT
from config.paths_config import DATALOADERS_DIR, DATASETS_DIR, PATCHES_DIR


def main():
    """
    Main validation function to check data pipeline integrity.
    """
    print("🔍 Validating data pipeline...")
    
    # Validate patches
    validate_patches()
    
    # Validate datasets
    validate_datasets()
    
    # Validate dataloaders
    validate_dataloaders()
    
    print("✅ Data validation completed!")


def validate_patches():
    """Validate patch data integrity."""
    print("  Validating patches...")
    
    patches_dir = PATCHES_DIR
    if not patches_dir.exists():
        print(f"    ❌ Patches directory not found: {patches_dir}")
        return
    
    # Check for patch files
    patch_files = list(patches_dir.glob("*.pkl"))
    if not patch_files:
        print("    ⚠️  No patch files found")
    else:
        print(f"    ✅ Found {len(patch_files)} patch files")


def validate_datasets():
    """Validate dataset integrity."""
    print("  Validating datasets...")
    
    datasets_dir = DATASETS_DIR
    if not datasets_dir.exists():
        print(f"    ❌ Datasets directory not found: {datasets_dir}")
        return
    
    # Check for dataset files
    dataset_files = list(datasets_dir.glob("*.pkl"))
    if not dataset_files:
        print("    ⚠️  No dataset files found")
    else:
        print(f"    ✅ Found {len(dataset_files)} dataset files")


def validate_dataloaders():
    """Validate dataloader integrity."""
    print("  Validating dataloaders...")
    
    dataloaders_dir = DATALOADERS_DIR
    if not dataloaders_dir.exists():
        print(f"    ❌ Dataloaders directory not found: {dataloaders_dir}")
        return
    
    # Check for dataloader files
    loader_files = list(dataloaders_dir.glob("*.pt"))
    if not loader_files:
        print("    ⚠️  No dataloader files found")
    else:
        print(f"    ✅ Found {len(loader_files)} dataloader files")
        
        # Test loading a dataloader
        try:
            test_loader = torch.load(dataloaders_dir / "small_train_loader.pt")
            print("    ✅ Successfully loaded test dataloader")
        except Exception as e:
            print(f"    ❌ Failed to load test dataloader: {e}")


if __name__ == "__main__":
    main() 