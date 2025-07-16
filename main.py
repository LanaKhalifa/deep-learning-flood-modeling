from config import prjs_metadata
from simulations_to_samples.scripts.generate_patches import process_project
from simulations_to_samples.scripts.generate_datasets import create_and_save_datasets
from simulations_to_samples.scripts.generate_dataloaders import create_and_save_dataloaders
from multi_architecture_training.A_train_all_archs_on_small_set.train import run_train_all_on_small

print("🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷 STAGE 1: FROM SIMULATIONS TO SAMPLES 🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷")

print("=" * 60)
print("🔷 Starting Flood Patch Generation... 🔷")
print("=" * 60)

"""
# Step 1: Process each project to produce patches
for prj_num, (prj_name, plans) in prjs_metadata.items():
    process_project(prj_num, prj_name, plans)
"""

print("=" * 60)
print("🔷 Starting Datasets Generation... 🔷")
print("=" * 60)

"""
# Step 2: Create and save datasets
create_and_save_datasets()
"""

print("=" * 60)
print("🔷 Starting Dataloaders Generation... 🔷")
print("=" * 60)

"""
# Step 3: Create and save dataloaders
create_and_save_dataloaders()
"""

print("🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷 STAGE 2: TRAIN DEEP LEARNING MODELS 🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷🔷")

# =============================
# STEP 1: Train All Architectures on Small Set
# =============================
print("=" * 60)
print("🔷 Training All Architectures on Small Set... 🔷")
print("=" * 60)

run_train_all_on_small()
