
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
### main.generate_patches()
 takes each simulation (210 in total) terrain and water depth maps and generates patches in `simulations_to_samples/processed_data/patches/'
<img width="1280" height="720" alt="terain_water_patching" src="https://github.com/user-attachments/assets/e4b21d2e-0dc8-4302-b20d-1bf7d61c178f" />





