
# 🌊 Deep Learning Flood Modeling

This repository provides a complete pipeline for emulating hydrodynamic flood simulations using deep learning and a closure iterative model.
The pipeline is organized into three stages:

1. **From Simulations to Data**: Transforms HEC-RAS (hydrodynamic simulation software) outputs and terrain inputs (.hdf and terrain .tif files) into patch-based deep learning-ready datasets of augmented terrain and water depth patches.
   - Directory: `simulations_to_samples/` Includes patch extraction, dataset, and dataloader generation scripts.
   - Before running: Move the large folder `hecras_simulations_results/` (shared via OneDrive) into: `simulations_to_samples/raw_data/`. This is necessary due to GitHub's file size limitations. With this, you can run the full pipeline.

2. **Training and Validation:** Trains modified existing deep learning models as well as custom desigend ones to predict water depth at the patch level
   - Directory: `multi_architecture_training/` Contains model architectures, training, and evaluation scripts.

3. **Closure Model**: scale patch predictions to coherent full-domain predictions. 
   - Directory: `full_domain_closure_best_model/` Includes utilities to map patch predictions back to the simulation domain grid.

## 1. From Simulations to Data
Note: “Plan” and “Simulation” are used interchangeably throughout this repository.

**main.generate_patches():** takes each simulation (210 in total) terrain and water depth maps and generates patches:
<img width="1280" height="366" alt="image" src="https://github.com/user-attachments/assets/066520cc-c46a-41b2-a808-cc0b7dfc524a" />

**main.generate_datasets()**:loads the extracted patches and assembles them into datasets  according to the following logic:
- small_train / small_val: Selects 2 simulations from each project. A project refers to a collection of simulations (i.e., a set of flood scenarios occuring on nearby terrains).
- big_train / big_val: Includes all simulations from all projects, excluding 7 simulations per project which are reserved for big_test.
- prj_03_train_val / prj_03_test: Mirrors the prj_03 simulations found in big_train, big_val, and big_test. prj_03 contains the highest-quality, hand-curated simulations—unlike the automatically generated settings used elsewhere in the dataset.

**main.generate_dataloaders():** simply generate deep learning ready dataloaders from datasets.
