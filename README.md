# 🌊 Deep Learning Flood Modeling

This repository implements a full pipeline for emulating hydrodynamic flood simulations using deep learning, followed by an iterative closure model for domain-wide prediction. The pipeline is organized into four main stages, each represented by a top-level directory:

| Section | Summary |
|--------|---------|
| [Installation](#-installation) | Setup instructions for conda environment, data download, quick test, and running from start to end |
| [1. From Simulations to Dataloaders](#1-from-simulations-to-dataloaders) | Converts HEC-RAS simulation outputs and terrain files (`.hdf` and terrain `.tif`) into deep learning-ready datasets composed of augmented terrain and water depth patches.|
| [2. Training Workflow](#2-training-workflow) | Trains multiple architectures including custom designed ones |
| [3. Evaluation and Visualization](#3-evaluation-and-visualization) | Evaluates the best model using a proposed metric and visualizes its predictions. |
| [4. Closure Model](#4-closure-model) | Uses an iterative closure model to upscale patch predictions to full-domain flood maps. |

---

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/LanaKhalifa/deep-learning-flood-modeling.git
   cd deep-learning-flood-modeling
   ```

2. **Create the conda environment:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate flood_env
   ```

4. **Download and setup data:**
   - Download the `hecras_simulations_results/` folder from the provided [OneDrive link](https://technionmail-my.sharepoint.com/:f:/g/personal/lana_khalifa_campus_technion_ac_il/EnIUbbko4yRFlVm4kEkRYj4Bj3CIg-xPNThrl7-OTetgHA?e=qvekae)
   - Move it to `simulations_to_samples/raw_data/hecras_simulations_results/`
   - This folder contains the HEC-RAS simulation outputs (.hdf files) and terrain data (.tif files) required for the pipeline.
  
3. **Quick test:**
   ```bash
   python main.py generate_patches
   ```
4. **Running from start to end:**
   ```bash
   python main.py run_all 
   ```
---
## 1. From Simulations to Dataloaders

> Note: “Plan” and “Simulation” are used interchangeably throughout this repository. A project (abbreviated as `prj`) refers to a collection of flood simulations conducted on remotely sensed terrains, all cut via QGIS from a single project unit defined by the U.S. Geological Survey's 3D Elevation Program (3DEP). Overall, there are 4 projects (`prj_03`, `prj_04`, `prj_05`, and `prj_06`) from which 210 simulations were run, differing from each other in terrain and flood event (water flow).

### How to Run:

```bash
python main.py generate_patches            #  saves patches to └── `simulations_to_samples/processed_data/patches_per_simulation/prj_##/plan_##/`
python main.py generate_datasets           #  saves dataset to └── `simulations_to_samples/processed_data/datasets/`
python main.py generate_dataloaders        #  saves dataloaders └── `simulations_to_samples/processed_data/dataloaders/`   
```

### **generate_patches**

Processes each simulation’s terrain and water depth maps by overlaying a patch grid and a dual grid (offset by half the patch size) to extract localized patches. These patches (each 32x32) are then augmented (flipping and rotating in various directions/degrees) and cleaned.

<img width="1280" height="373" alt="image" src="https://github.com/user-attachments/assets/cc712de5-0266-44e5-bde7-b5519997ad93" />

### **generate_datasets**

Loads the patches of each simulation and assembles them into datasets as follows:

| Name                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `small_train` / `small_val`   | 7 simulations from `prj_03`, used for rapid experimentation and model comparison. |
| `big_train` / `big_val`       | All simulations across all projects, excluding 7 per project (reserved for `big_test`). |
| `prj_03_train_val` / `prj_03_test` | Mirrors `big_*` sets for `prj_03`, which contains hand-curated simulations. Useful for benchmarking model performance on high-quality data. |

### **generate_dataloaders**

Generates deep learning-ready dataloaders from the dataset. Each sample follows the format below. (Note: The downsampler is a learnable module that reduces the terrain input from 321×321 to 32×32, allowing it to be concatenated along the channel dimension with the 32×32 water depth maps.)

<img width="980" height="377" alt="image" src="https://github.com/user-attachments/assets/981097c6-b6da-4b15-986a-6e5d445e38e6" />

See `📁 simulations_to_samples/processed_data/dataloaders/figures/` for random samples from each dataloader. Here are 6 samples:

<img width="1280" height="497" alt="image" src="https://github.com/user-attachments/assets/35cd3e72-ee02-4ca6-83d8-244fe91dd711" />

---

## 2. Training Workflow
This stage involves systematic experimentation with various architectures to predict future water depth at the patch level. The process is organized into subdirectories within 📁 `multi_architecture_training/`:

| Folder                             | Purpose |
|------------------------------------|---------|
| 📁 `A_tune_one_arch_on_small_set/`    | Fine-tunes the proposed architecture (*Non-Downsampling Convolutions with Self-Attention*, along with the *Alternating Stride Downsampler*) and explores database, architectural, and optimization settings using `small_train` / `small_val`. |
| 📁 `B_train_all_archs_on_small_set/`  | Trains multiple architectures on `small_train` to position the proposed model within a broader landscape. Identifies top candidates for large-scale training. |
| 📁 `C_train_best_four_on_big_set/`    | Retrains the top four architectures on `big_train` / `big_val` to identify the best model. |

### How to Run
```bash
python main.py calculate_dummy_losses     # Baseline L1 losses using steady-state dummy model 
                                          # saves losses at └── multi_architecture_training/training_utils/dummy_small_val_loss.pt or dummy_big_val_loss.pt

python main.py A_train                    # Train using config A (similarly B or C)
                                          # saves trained model at └── A_*/saved_trained_models/Arch_##/model.pth
                                          # saves losses at └── A_*/saved_losses/Arch_##/losses.pt

python main.py A_plot_losses              # Plot training and validation losses for config A (similarly B or C) 
                                          # saves learning curves at └── A_*/figures/learning_curves.png
```

### Shared Utilities (A, B, and C)

- Models implemented in: `/multi_architecture_training/models/`  
- Configuration file: `/config/model_configs.py`
- Train file: `/multi_architecture_training/training_utils/train_model.py`

### B_plot_losses and C_plot_losses
Plots L1 training and validation loss curves for all architectures trained on `small_train` (Stage B) and for the top four architectures retrained on `big_train` (Stage C).

<p align="center">
  <img src="https://github.com/user-attachments/assets/bb3c06e3-29e6-4607-b658-488b3b2bbcb1" alt="learning_curves_aa" style="width: 66%;" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ca1eea6-00e7-43ce-9da2-2e3227171dea" alt="learning_curves_cc" style="width: 66%;" />
</p>

---

## 3. Evaluation and Visualization

> **Note:** RAE measures how much error the trained model makes compared to a baseline dummy model which predicts no change in water depth. Specifically, it computes the total absolute difference between the model’s predictions and the ground truth, divided by the total absolute difference between the dummy model’s predictions and the ground truth.

### How to Run

```bash
python main.py generate_rae_boxplots                      # save to  └── `evaluate_and_visualize_best_model/boxplots/arch_05_rae_boxplots.png`
python main.py plot_entire_batch_predictions              # save to └── `evaluate_and_visualize_best_model/visual_predictions/ten_predictions_{idx}.png` 
```

### generate_rae_boxplots

<p align="center">
  <img src="https://github.com/user-attachments/assets/51979547-30fa-4954-983f-543d1c05a06a" alt="arch_05_rae_boxplots" style="width: 66%;" />
</p>

### plot_entire_batch_predictions

Plots the predictions of the best model on 300 random samples from `prj_03_test` set. Each figure includes 10 samples, resulting in a total of 30 figures. See `📁 evaluate_and_visualize_best_model/visual_predictions`. Here is one figure:

<p align="center">
  <img src="https://github.com/user-attachments/assets/02eae530-8fad-4451-aa62-68331ffb3087" alt="ten_samples_1" style="width: 30%;" />
</p>

___
## 4. Closure Model
> **Note:** The closure model implementation is currently not available as it is under review for publication. This section will be updated upon article acceptance. In the meantime, you can view the results in this OneDrive folder: Closure Model Results - [OneDrive](https://technionmail-my.sharepoint.com/:f:/g/personal/lana_khalifa_campus_technion_ac_il/ElScJxuuHdZGug4VUbb-ehwB3aJaE-8nPgitDCgEBfdvZg?e=BCVYo8)
> 
> The folder contains:
> - **9_Maps**: Shows the predicted water depth and the ground truth over the entire domain, compared to a dummy model. Each subplot x and y axes show the number of pixels this flood is modeled over. Each pixel is 10m in simulation.  
> - **Converging**: Shows the MAE between the prediction and ground truth as the solution evolves until it converges.  
>  
> These figures were generated from test simulations used to create test loaders. Each PNG is named using the format {prj_num}_{plan_num}_{time}.png, referring to the project, plan, and the input time t_initial when the closure model was applied.  
>  
> Below are selected examples of the 9 maps with their corresponding convergence curves:

<img width="5760" height="3960" alt="111_prj_03_plan_71_t_70" src="https://github.com/user-attachments/assets/5dde5ac4-521e-451c-810a-a0b3a8d1da4a" />
<img width="2160" height="720" alt="111c_prj_03_plan_71_t_70" src="https://github.com/user-attachments/assets/2bf82667-f07c-4dbd-83a4-f9ca646b7f98" />
<img width="5760" height="3960" alt="222_prj_04_plan_44_t_210" src="https://github.com/user-attachments/assets/291b4ff1-50fa-4d37-acd1-fa2a4fed4ad4" />
<img width="2160" height="720" alt="222c_prj_04_plan_44_t_210" src="https://github.com/user-attachments/assets/2c7ad89c-3b63-4765-a67f-59e379b9a6d6" />

