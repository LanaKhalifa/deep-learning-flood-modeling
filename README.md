# 🌊 Deep Learning Flood Modeling

This repository implements a full pipeline for emulating hydrodynamic flood simulations using deep learning, followed by an iterative closure model for domain-wide prediction. The pipeline is structured into three main stages, each represented by a top-level directory in the repository:

1. **📁 simulations_to_data/**: Transforms HEC-RAS (hydrodynamic simulation software) outputs and terrain inputs (.hdf and terrain .tif files) into patch-based, deep learning-ready datasets of augmented terrain and water depth patches.  
   - **Directory**: Includes patch extraction, dataset, and dataloader generation scripts.  
   - **Before running**: Move the large folder `hecras_simulations_results/` (shared via OneDrive) into `simulations_to_samples/raw_data/`. This is necessary due to GitHub's file size limitations. With this, you can run the full pipeline from start to end.

2. **📁 multi_architecture_training/**: Trains custom-designed deep learning models, as well as modified models from the literature, to predict water depth at the patch level.  
   - **Directory**: Includes model architectures, training, and evaluation scripts.

3. **📁 full_domain_closure_best_model/**: Scales patch predictions to coherent full-domain predictions using the best architecture from the previous step.

---

## 1. From Simulations to Dataloaders

> Note: “Plan” and “Simulation” are used interchangeably throughout this repository. A project (abbreviated as `prj`) refers to a collection of flood simulations conducted on remotely sensed terrains, all cut via QGIS from a single project unit defined by the U.S. Geological Survey's 3D Elevation Program (3DEP). Overall, there are 4 projects (`prj_03`, `prj_04`, `prj_05`, and `prj_06`) from which 210 simulations were run, differing from each other in terrain and flood event (water flow).

### How to Run

```bash
python main.py generate_patches 
python main.py generate_datasets           
python main.py generate_dataloaders         
```

### Output Paths

- `generate_patches` saves to:  
  └── `simulations_to_samples/processed_data/patches_per_simulation/prj_##/plan_##/`

- `generate_datasets` saves to:  
  └── `simulations_to_samples/processed_data/datasets/`

- `generate_dataloaders` saves to:  
  └── `simulations_to_samples/processed_data/dataloaders/`

### **generate_patches**

Processes each simulation’s terrain and water depth maps by overlaying a patch grid and a dual grid (offset by half the patch size) to extract localized patches. These patches are then augmented (e.g., flipping and rotating in various degrees) and cleaned.

<img width="1280" height="373" alt="image" src="https://github.com/user-attachments/assets/cc712de5-0266-44e5-bde7-b5519997ad93" />

### **generate_datasets**

Loads the patches from each simulation and assembles them into datasets as follows:

| Name                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `small_train` / `small_val`   | 7 simulations from `prj_03`, used for rapid experimentation and model comparison. |
| `big_train` / `big_val`       | All simulations across all projects, excluding 7 per project (reserved for `big_test`). |
| `prj_03_train_val` / `prj_03_test` | Mirrors `big_*` sets for `prj_03`, which contains hand-curated simulations. Useful for benchmarking model performance on high-quality data. |

### **generate_dataloaders**

Generates deep learning-ready dataloaders from datasets. Each sample should look as follows (ignore the downsampler part for now):

<img width="980" height="377" alt="image" src="https://github.com/user-attachments/assets/981097c6-b6da-4b15-986a-6e5d445e38e6" />

See `📁 simulations_to_samples/processed_data/dataloaders/figures/` for random samples from each dataloader. Here are 6 samples:

<img width="1280" height="497" alt="image" src="https://github.com/user-attachments/assets/35cd3e72-ee02-4ca6-83d8-244fe91dd711" />

---

## 2. Training and Evaluation Workflow
This stage involves systematic experimentation with various architectures to predict future water depth at the patch level. The process is organized into subdirectories within:

📁 `multi_architecture_training/`

| Folder                             | Purpose |
|------------------------------------|---------|
| 📁 `A_train_all_archs_on_small_set/`  | Trains multiple architectures on `small_train` to position the custom-designed model (*Non-Downsampling Convolutions with Self-Attention*) within a broader landscape of adapted models and evaluate the feasibility of the downsampler branch (*Alternating Stride Downsampler*). Also identifies top-performing candidates worth training on `big_train`. |
| 📁 `B_tune_one_arch_on_small_set/`   | Fine-tunes the proposed architecture (*Arch_04*) on `small_train` / `small_val`. |
| 📁 `C_train_best_four_on_big_set/`   | Retrains the top four architectures on `big_train` / `big_val`. |

### How to Run

```bash
python main.py calculate_dummy_losses  # Baseline L1 losses using steady-state dummy model
python main.py X_train                 # Replace X with A, B, or C to train
python main.py X_plot_losses           # Replace X with A, B, or C to plot losses
```

### Output Paths

- `calculate_dummy_losses` saves to:  
  └── `simulations_to_samples/training_utils/dummy_small_val_loss.pt`

- `X_train` saves to:  
  └── `X_*/saved_trained_models/Arch_##/model.pt`  
  └── `X_*/saved_losses/Arch_##/losses.pt`

- `X_plot_losses` saves to:  
  └── `X_*/learning_curves.png`

### Shared Utilities (A, B, and C)

- Models implemented in: `multi_architecture_training/models/`  
- Configuration file: `multi_architecture_training/config/model_configs.py`
- Train file: 'multi_architecture_training/training_utils/train_model.py'

### A_plot_losses and C_plot_losses

Generate visualizations of the L1 training and validation loss curves for all architectures trained on `small_train` (Stage A) and for the top four architectures retrained on `big_train` (Stage C).

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb3c06e3-29e6-4607-b658-488b3b2bbcb1" alt="learning_curves_aa" style="width: 66%;" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ca1eea6-00e7-43ce-9da2-2e3227171dea" alt="learning_curves_cc" style="width: 66%;" />
</p>

---

## 3. Evaluation and Visualization

> **Note:** RAE measures how much error the trained model makes compared to a baseline dummy model. Specifically, it computes the total absolute difference between the model’s predictions and the ground truth, divided by the total absolute difference between the dummy model’s predictions and the ground truth.

### How to Run

```bash
python main.py generate_rae_boxplots
python main.py plot_entire_batch_predictions
```

### Output Paths

- `generate_rae_boxplots` saves to:  
  └── `evaluate_and_visualize_best_model/boxplots/arch_05_rae_boxplots.png`

- `plot_entire_batch_predictions` saves to:  
  └── `evaluate_and_visualize_best_model/visual_predictions/ten_predictions_{idx}.png` 


### generate_rae_boxplots

<p align="center">
  <img src="https://github.com/user-attachments/assets/51979547-30fa-4954-983f-543d1c05a06a" alt="arch_05_rae_boxplots" style="width: 66%;" />
</p>

### plot_entire_batch_predictions

Plots the predictions of the best model on one batch (300 samples) from `prj_03_test` set. Each figure includes 10 samples from one batch, resulting in a total of 30 figures. See `📁 evaluate_and_visualize_best_model/visual_predictions` for all samples in first batch. Here are 10 samples:

<p align="center">
  <img src="https://github.com/user-attachments/assets/02eae530-8fad-4451-aa62-68331ffb3087" alt="ten_samples_1" style="width: 30%;" />
</p>
I've cherry picked intricate predictions on test set to show the model's capacity: 

___
## 4. Closure Model

