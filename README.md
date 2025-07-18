
# 🌊 Deep Learning Flood Modeling

This repository implements a full pipeline for emulating hydrodynamic flood simulations using deep learning, followed by an iterative closure model for domain-wide prediction. The pipeline is structured into three main stages, each represented by a top-level directory in the repository:

1. **📁 simulations_to_data/**: Transforms HEC-RAS (hydrodynamic simulation software) outputs and terrain inputs (.hdf and terrain .tif files) into patch-based deep learning-ready datasets of augmented terrain and water depth patches.
   - Directory: Includes patch extraction, dataset, and dataloader generation scripts.
   - Before running: Move the large folder `hecras_simulations_results/` (shared via OneDrive) into: `simulations_to_samples/raw_data/`. This is necessary due to GitHub's file size limitations. With this, you can run the full pipeline.

2. **📁 multi_architecture_training/**: Trains custom designed deep learning models as well as modified ones from the literature to predict water depth at the patch level
   - Directory: Includes model architectures, training, and evaluation scripts.

3. **📁 full_domain_closure_best_mosel/**: scales patch predictions to coherent full-domain predictions. utilizes the best architecture from the previous step. 
### Setup: 
## 1. From Simulations to Dataloaders
Note: “Plan” and “Simulation” are used interchangeably throughout this repository. A project (abbreviated as `prj`) refers to a collection of flood simulations conducted on remotely sensed terrains, all cut via QGIS from a single project unit defined by the U.S. Geological Survey's 3D Elevation Program (3DEP). overall there are 4 projects (prj_03, prj_04, prj_05 and prj_06) from which 210 simulations were run, differing from each other with terrain and flood event (water flow). 

**Output Paths:**
- `generate_patches` saves to:  
  └── `simulations_to_samples/processed_data/patches_per_simulation/prj_##/plan_##/`

- `generate_datasets` saves to:  
  └── `simulations_to_samples/processed_data/datasets/`

- `generate_dataloaders` saves to:  
  └── `simulations_to_samples/processed_data/dataloaders/`

### **main.generate_patches:** 
takes each simulation's terrain and water depth maps and generates preprocessed patches after augmentation and cleaning:
<img width="1280" height="373" alt="image" src="https://github.com/user-attachments/assets/cc712de5-0266-44e5-bde7-b5519997ad93" />

### **main.generate_datasets**: 
loads the patches from each simulation and assembles them into datasets as follows:

| Name                     | Description |
|--------------------------|-------------|
| `small_train` / `small_val` | Selects 7 simulations from prj_03. Used for fast experimentation and architectural comparison. |
| `big_train` / `big_val`     | Includes all simulations from all projects, **excluding 7 per project**, which are reserved for `big_test`. |
| `prj_03_train_val` / `prj_03_test` | Subset of simulations from `prj_03` mirroring `big_*` sets samples. Unlike the other projects, prj_03 contains hand-curated simulations rather than those generated through automated processes. Evaluating performance on this dataset is therefore essential.|

### **main.generate_dataloaders:** 
generates deep learning ready dataloaders from datasets. each sample should look as follows (ignore the downsampler part for now):

<img  width="980" height="377" alt="image" src="https://github.com/user-attachments/assets/981097c6-b6da-4b15-986a-6e5d445e38e6" />

see 📁 `simulations_to_samples/processed_data/dataloaders/figures/` to see random samples from each dataloader. here are 6 samples:
<img width="1280" height="497" alt="image" src="https://github.com/user-attachments/assets/35cd3e72-ee02-4ca6-83d8-244fe91dd711" />

## 2. Training the Proposed Architecture 
This stage involves systematic experimentation with various architectures to predict future water depth at the patch level. The process is organized into subdirectories within:

> 📁 `multi_architecture_training/`

Each subdirectory reflects a distinct phase in the research workflow:

| Directory | Description |
|-----------|-------------|
| 📁`A_train_all_archs_on_small_set/` | Trains multiple architectures on `small_train` to position the custom-designed model (*Non-Downsampling Convolutions with Self-Attention*) within a broader landscape of adapted models and evaluate the feasibility of the downsampler branch (*Alternating Stride Downsampler*). Also identifies top-performing candidates worth training on `big_train`.|
| 📁`B_tune_one_arch_on_small_set/` | Fine-tunes the proposed model (*Non-Downsampling Convolutions with Self-Attention*) on `small_train` / `small_val`.|
| 📁`C_train_best_three_on_big_set/` | Retraining the three best architectures on the full dataset (`big_train` / `big_val`) to assess scalability and robustness. |
| 📁`D_boxplots_RAE_all_sets/` | Evaluation of the final models using Relative Absolute Error (RAE) across all datasets. |
| 📁`D_visualize_prediction_and_errors_test_set/` | Visual comparison between predicted and ground truth water depths on the test set. |
| 📁`D_compare_all_archs_runtime_size_performance/` | Summary of model inference time, parameter count, and overall performance to support trade-off analysis. |

> 📁`A_train_all_archs_on_small_set/`
### 2.A: Training All Architectures on Small Set
#### main.A_train:
Trains the models defined in:  
└── `multi_architecture_training/models/`  
using configuration settings from:  
└── `multi_architecture_training/A_train_all_archs_on_small_set/architecture_configs.py`

#### main.A_plot_losses:
└──  `multi_architectures_training/A_train_all_archs_on_small_set/learning_curves.png`: 
<img width="2742" height="1114" alt="image" src="https://github.com/user-attachments/assets/81a21afc-a4cd-4dd5-b298-4fd6d097ee60" />

> 📁`B_tune_one_arch_on_small_set/`
### 2.B: Tuning Non-Downsampling Convolution with Self-Attention
Note: This phase involved extensive experimentation to fine-tune the proposed model. Only the final, optimized version is included and used in this repository.
#### main.B_train
Trains arch_04 defined in:  
└── `multi_architecture_training/models/`  
using configuration settings from:  
└── `multi_architecture_training/B_tune_one_arch_on_small_set/arch_04_tuned_config.py`
#### main.B_plot_losses
└── `multi_architecture_training/B_tune_one_arch_on_small_set/learning_curves.py`

<p align="center">
<img width="400" height="333" alt="image" src="https://github.com/user-attachments/assets/00d93142-c735-477b-8eca-92a54dae3b10" />
</p>

> 📁`C_train_best_on_big_set/`


