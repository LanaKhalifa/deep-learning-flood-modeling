from config import prjs_metadata
from simulations_to_samples.scripts.generate_patches import process_project
from simulations_to_samples.scripts.generate_datasets import create_and_save_datasets

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

# Step 2: Create and save datasets
create_and_save_datasets()


