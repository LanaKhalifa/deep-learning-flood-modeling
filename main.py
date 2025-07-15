from config import prjs_metadata
from simulations_to_samples.scripts.generate_patches import process_project

print("=" * 60)
print("🔷 Starting Flood Patch Generation Pipeline... 🔷")
print("=" * 60)

# Step 1: Process each project
for prj_num, (prj_name, plans) in prjs_metadata.items():
    process_project(prj_num, prj_name, plans)

"""
# Step 2: Concatenate train+val sets across all projects
concatenate_all_train_val_chunks(prj_sublists)

# Step 3: Concatenate test sets across all projects
concatenate_all_test_chunks(prj_sublists)

# Step 4: Concatenate project 03 train+val+test split separately
concatenate_prj03_train_val_test_chunks()

# Step 5: Create small subset
generate_small_train_val_subset()

print("All patch datasets have been generated and saved.")
"""
