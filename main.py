from config import prj_sublists
from generate_database import (process_project,
                              concatenate_all_train_val_chunks,
                              concatenate_all_test_chunks,
                              concatenate_prj03_train_val_test_chunks,
                              generate_small_train_val_subset)

print("Starting full flood patch generation pipeline...")

# Step 1: Process each project and save patches in chunks of data due to memory limitation 
for prj_num, (prj_name, sublists) in prj_sublists.items():
    process_project(prj_num, prj_name, sublists)

# Step 2: Concatenate train+val sets across all projects
concatenate_all_train_val_chunks(prj_sublists)

# Step 3: Concatenate test sets across all projects
concatenate_all_test_chunks(prj_sublists)

# Step 4: Concatenate project 03 train+val+test split separately
concatenate_prj03_train_val_test_chunks()

# Step 5: Create small subset
generate_small_train_val_subset()

print("All patch datasets have been generated and saved.")
