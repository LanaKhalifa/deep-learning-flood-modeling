
# 🌊 Deep Learning Flood Modeling

This repository implements a full pipeline for emulating hydrodynamic flood simulations using deep learning, followed by an iterative closure model for domain-wide prediction. The pipeline is structured into three main stages, each represented by a top-level directory in the repository:

1. **📁 simulations_to_data/**: Transforms HEC-RAS (hydrodynamic simulation software) outputs and terrain inputs (.hdf and terrain .tif files) into patch-based deep learning-ready datasets of augmented terrain and water depth patches.
   - Directory: Includes patch extraction, dataset, and dataloader generation scripts.
   - Before running: Move the large folder `hecras_simulations_results/` (shared via OneDrive) into: `simulations_to_samples/raw_data/`. This is necessary due to GitHub's file size limitations. With this, you can run the full pipeline.

2. **📁 multi_architecture_training/**: Trains modified existing deep learning models as well as custom desigend ones to predict water depth at the patch level
   - Directory: Includes model architectures, training, and evaluation scripts.

3. **📁 full_domain_closure_best_mosel/**: scale patch predictions to coherent full-domain predictions. utilizes the best architectures from the previous step. 
### Setup: 
## 1. From Simulations to Dataloaders
Note: “Plan” and “Simulation” are used interchangeably throughout this repository. A *project* (abbreviated as `prj`) refers to a collection of flood simulations conducted on nearby terrains, all extracted from a single project unit defined by the U.S. Geological Survey's 3D Elevation Program (3DEP).

**Output Paths:**
- `generate_patches` saves to:  
  └── `simulations_to_samples/processed_data/patches_per_simulation/PRJ_##/PLAN_##/`

- `generate_datasets` saves to:  
  └── `simulations_to_samples/processed_data/datasets/`

- `generate_dataloaders` saves to:  
  └── `simulations_to_samples/processed_data/dataloaders/`

### **main.generate_patches:** 
takes each simulation (210 in total) terrain and water depth maps and generates and augments patches:
<img width="1280" height="366" alt="image" src="https://github.com/user-attachments/assets/066520cc-c46a-41b2-a808-cc0b7dfc524a" />

### **main.generate_datasets**: 
loads the patches from each simulation and assembles them into datasets as follows:

| Name                     | Description |
|--------------------------|-------------|
| `small_train` / `small_val` | Selects 2 simulations from each project. Used for fast experimentation and architectural comparison. |
| `big_train` / `big_val`     | Includes all simulations from all projects, **excluding 7 per project**, which are reserved for `big_test`. |
| `prj_03_train_val` / `prj_03_test` | Subset of simulations from `prj_03` already included in the `big_*` sets. `prj_03` consists of hand-curated simulations, unlike the automatically generated ones in other projects. |

### **main.generate_dataloaders:** 
simply generate deep learning ready dataloaders from datasets. each sample should look as follows:
<img width="920" height="377" alt="image" src="https://github.com/user-attachments/assets/981097c6-b6da-4b15-986a-6e5d445e38e6" />

## 2. Training and Validating Multiple Architectures

This stage involves systematic experimentation with various neural network architectures to predict water depth at the patch level.  
The process is organized into clearly defined subdirectories within:

> 📁 `multi_architecture_training/`

Each subdirectory reflects a distinct phase in the research workflow:

| Directory | Description |
|-----------|-------------|
| `A_train_all_archs_on_small_set/` | Initial comparison of multiple architectures trained on a small dataset. Used to identify high-performing candidates. |
| `B_tune_one_arch_on_small_set/` | Hyperparameter tuning of the top-performing model (`Arch_04`) on the same small dataset. |
| `C_train_best_three_on_big_set/` | Retraining the three best architectures on the full dataset (`big_train` / `big_val`) to assess scalability and robustness. |
| `D_boxplots_RAE_all_sets/` | Evaluation of the final models using Relative Absolute Error (RAE) across all datasets. |
| `D_visualize_prediction_and_errors_test_set/` | Visual comparison between predicted and ground truth water depths on the test set. |
| `D_compare_all_archs_runtime_size_performance/` | Summary of model inference time, parameter count, and overall performance to support trade-off analysis. |

