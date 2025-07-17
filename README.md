
# 🌊 Deep Learning Flood Modeling

This repository provides a complete pipeline for emulating hydrodynamic flood simulations using deep learning and a closure iterative model.
The pipeline is organized into three stages, which are saved as directories in the repository's root: 

1. **📁 simulations_to_data**: Transforms HEC-RAS (hydrodynamic simulation software) outputs and terrain inputs (.hdf and terrain .tif files) into patch-based deep learning-ready datasets of augmented terrain and water depth patches.
   - Directory: `simulations_to_samples/` Includes patch extraction, dataset, and dataloader generation scripts.
   - Before running: Move the large folder `hecras_simulations_results/` (shared via OneDrive) into: `simulations_to_samples/raw_data/`. This is necessary due to GitHub's file size limitations. With this, you can run the full pipeline.

2. **📁 multi_architecture_training/** Trains modified existing deep learning models as well as custom desigend ones to predict water depth at the patch level
   - Directory: `multi_architecture_training/` Contains model architectures, training, and evaluation scripts.

3. **📁 full_domain_closure_best_mosel**: scale patch predictions to coherent full-domain predictions. utilizes the best architectures from the previous step. 
   - Directory: `full_domain_closure_best_model/` Includes utilities to map patch predictions back to the simulation domain grid.
### Setup: 
## 1. From Simulations to Dataloaders
Note: “Plan” and “Simulation” are used interchangeably throughout this repository.

**Output Paths:**
- `generate_patches` saves to:  
  └── `simulations_to_samples/processed_data/patches_per_simulation/PRJ_##/PLAN_##/`

- `generate_datasets` saves to:  
  └── `simulations_to_samples/processed_data/datasets/`

- `generate_dataloaders` saves to:  
  └── `simulations_to_samples/processed_data/dataloaders/`

**python main.generate_patches:** takes each simulation (210 in total) terrain and water depth maps and generates and augments patches:
<img width="1280" height="366" alt="image" src="https://github.com/user-attachments/assets/066520cc-c46a-41b2-a808-cc0b7dfc524a" />

**python main.generate_datasets**: loads the patches from each simulation and assembles them into datasets as follows:
- small_train / small_val: Selects 2 simulations from each project. A project refers to a collection of simulations (i.e., a set of flood scenarios occuring on nearby terrains).
- big_train / big_val: Includes all simulations from all projects, excluding 7 simulations per project which are reserved for big_test.
- prj_03_train_val / prj_03_test: Mirrors the prj_03 simulations found in big_train, big_val, and big_test. prj_03 contains the highest-quality, hand-curated simulations—unlike the automatically generated settings used elsewhere in the dataset.

**python main.generate_dataloaders:** simply generate deep learning ready dataloaders from datasets. each sample should look as follows:
<img width="920" height="377" alt="image" src="https://github.com/user-attachments/assets/981097c6-b6da-4b15-986a-6e5d445e38e6" />
- `generate_dataloaders` saves to:  
## 2. Training and Validating Multipe Architectures:
the steps in which the reserach was conducted, are shown in the correct order in multi_architecture_training/ directory at root of the project. 


Name	Last commit message	Last commit date
..
A_train_all_archs_on_small_set
B_tune_one_arch_on_small_set
C_train_best_three_on_big_set
D_boxplots_RAE_all_sets
D_visualize_prediction_and_errors_test_set
