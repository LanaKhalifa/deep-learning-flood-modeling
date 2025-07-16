<img width="1280" height="720" alt="fig_14" src="https://github.com/user-attachments/assets/dfdbd8d9-d1d6-49cd-b1e6-683e19b28708" />

remember to say that in stage 2, only the final configuration were were implemented to present a cleaner story. 

# 🌊 Deep Learning Flood Modeling

This repository provides a complete pipeline for emulating hydrodynamic flood simulations using deep learning.
The pipeline is organized into three stages:

1. **From Simulations to Data**  
   Transforms HEC-RAS (hydrodynamic simulation software) outputs into patch-based datasets of augmented terrain and water depth patches.
   - **Directory**: `simulations_to_samples/`  
     Includes patch extraction, dataset creation, and dataloader generation scripts.
   - **Before running**:  
     Move the large folder `hecras_simulations_results/` (shared via OneDrive) into:  
     `simulations_to_samples/raw_data/hecras_simulations_results/`  
     This is necessary due to GitHub's file size limitations. With this structure in place, you can run the full pipeline using `main.py`.

2. **Training and Validation**  
   Trains modified existing deep learning models as well as custom desigend ones to predict water depth at the patch level
   - **Directory**: `multi_architecture_training/`  
     Contains model architectures, training scripts, and evaluation logic.

3. **Closure Model**  
   scale patch predictions to coherent full-domain predictions. 
   - **Directory**: `full_domain_closure_best_model/`  
     Includes utilities to map patch predictions back to the simulation domain grid.


## From Simulations to Data
ℹNote: “Plan” and “Simulation” are used interchangeably throughout this repository.

This stage prepares deep learning-ready datasets from HEC-RAS simulation outputs (.hdf and terrain .tif files). The process includes:

1. Extracting Patches
Each HEC-RAS simulation (referred to as a plan in HEC-RAS terminology) is spatially split into multiple smaller patches:

Terrain patch: 321×321
Water depth patch at time t: 32×32
Water depth patch at time t+Δt: 32×32 

Here is an example for splitting 

<img width="1280" height="720" alt="fig_14" src="https://github.com/user-attachments/assets/644d3fd9-7f8f-4668-8025-1e1843113b07" />

2. Creating Datasets
Patches are grouped into larger datasets for training and testing. We define three types of datasets:

📦 big_* datasets

Combine patches from all projects (03, 04, 05, 06)
Exclude the last 7 simulations (plans) from each project, which are reserved for testing
Used to maximize training diversity
📦 prj_03_* datasets

Include only patches from Project 03 (hand-curated simulations)
Used to test model generalization on high-quality, minimally automated data
📦 small_* dataset

Contains only the first two simulations of each project
Designed for debugging, parameter tuning, and quick iteration
All datasets are stored as .pkl files under:

simulations_to_samples/processed_data/datasets/
3. Generating Dataloaders
Each dataset is transformed into a PyTorch DataLoader, providing inputs and labels formatted for model training:

Input (2 channels):
Channel 1: water depth at t+Δt with zeroed interior and preserved boundary
Channel 2: water depth at t
Auxiliary input: terrain patch
Label: water depth difference → depth_next - depth
The dataloaders are optionally split into training and validation and stored under:

simulations_to_samples/processed_data/dataloaders/
▶️ Run Full Pipeline
To generate all patches, datasets, and dataloaders:

python main.py
This will:

Extract patches from all simulations
Create dataset groupings
Build and save dataloaders
✅ Make sure to place your raw data folder hecras_simulations_results/ inside:
simulations_to_samples/raw_data/hecras_simulations_results/
(This is required due to GitHub’s file size limitations.)
Let me know when you're ready to proceed to the Training and Validation section.


Additional key files:
- `main.py` – Runs the full preprocessing pipeline.
- `config.py` – Central configuration for paths, patch size, and simulation metadata.
- `environment.yml` – Conda environment setup.
- `utils/` – Shared utility functions.



hyperparameter tuning of each architetcure was omitted as it is not clean. so stage 1 was omitted, and we straight beging with training all architecture on big_train_val. meaning trainig the different architecture. 

whenevr you want to download anything, simply do that by mocing form onedrive to there and then you can make any stage as you wish 





📊 Project Workflow Overview
From Simulations to Data
Extract terrain, depth, and future depth patches from HEC-RAS simulations and generate structured datasets and dataloaders.
Training and Validation
Train various deep learning architectures to predict water depth evolution using the generated patch datasets.

📘 README Structure

🌊 1. From Simulations to Data
What is a "plan" vs. a simulation
Extracting patches with PatchExtractorProcessor
Dataset design: big_, small_, prj_03_
Creating PyTorch DataLoaders
Directories and file formats
🧠 2. Training and Validation
Supported architectures: UNet, ResNet, Attention, GAN, etc.
Input/Output formulation: BC + depth → Δdepth
Loss functions used (e.g., L1, smooth L1)
Training pipeline overview (early stopping, LR scheduling)
Validation strategy and monitoring
Tips for architecture debugging






## 📦 How to Run

Clone the repository and create the environment:

```bash
git clone https://github.com/yourusername/deep-learning-flood-modeling.git
cd deep-learning-flood-modeling
conda env create -f environment.yml
conda activate HDF_env

🌊 Deep Learning Flood Modeling

This repository contains a complete pipeline for converting HEC-RAS flood simulation outputs into patch-based datasets suitable for deep learning training.

📁 Project Structure

├── main.py                       # Entry point: runs full pipeline
├── config.py                     # All project config and sublists
├── database_generator.py         # Preprocessing logic
├── PatchExtractorProcessor.py   # Core class: extract and process patches
├── generate_dataloaders.py      # (Optional) for training input loading
├── HECRAS_Simulations_Results/  # Raw input files (HDFs, TIFFs)
├── Databases/                   # Output patches (.pkl)
└── README.md

🚀 How to Run

Install dependencies

pip install -r requirements.txt  # ← or add your own dependencies

Run full preprocessing pipeline

python main.py

This will:

Process all simulation plans

Extract and augment patches

Save per-group chunks

Concatenate all training/validation/test splits

📦 Output Files

Saved under ./Database/:

saved_in_chunks/{project_id}/ → raw patches per sublist

big_train_val_depths.pkl, etc. → unified datasets

prj_03_train_val_depths.pkl → separate prj_03 split

small_train_val_depths.pkl → dev/debug subset

📐 Patch Format

Each training sample is a dict:

{
  'terrain': np.ndarray (321x321),
  'depth': np.ndarray (32x32),
  'depth_next': np.ndarray (32x32)
}

Augmentations include flipping + rotation.

🧠 Tips

Edit config.py to add/remove plans or change paths.

Intermediate HDFs and TIFFs go in HECRAS_Simulations_Results/

All constants (patch size, grid spacing, etc.) are configurable.

🧼 Cleanup

To remove macOS metadata files:

find . -name '__MACOSX' -or -name '.DS_Store' | xargs rm -rf

🧪 Coming Next

Training script

Dataloader module

Model evaluation code

📄 License

MIT License — free to use with attribution.

Databases

We have 3 main databases:

big_train_val and big_test: created from all simulations.
big_test patches come from simulations that were not used in training/validation.
prj_03_train_val and prj_03_test: extracted from project 03 only. Since this project was hand-curated, we wanted to evaluate performance on it separately.
small_train_val: used for parameter tuning, so that each training run is fast.
Whenever you see: 03, 04, 05, or 06, these refer to the 4 HEC-RAS projects used in this work.
Each project contains multiple simulations (called plans in HEC-RAS), and has its own terrain file(s) nearby.

Database Directory

Use a specific structure like:

./Database/chunks/{prj_num}/
This makes it clear that:

These are partial datasets
Each belongs to a specific project
They were saved per sublist chunk for memory efficiency
Project Structure

The project is composed of 3 main parts:

Preprocessing: Convert raw HEC-RAS simulation results (HDF5 and terrain files) into datasets of patches.
Training: Train deep learning models on the generated patches.
The from_HDF script is the first step in the preprocessing pipeline.

Data Availability

The folder HECRAS_Simulations_Results/ contains raw terrain and simulation files generated using HEC-RAS.
Due to GitHub file size limitations, these files are not included in the repository.

You can download the full dataset here:
Download Simulation Results

After downloading, place the folder named HECRAS_Simulations_Results in the main directory of the cloned repository — that is, in the same location as train.py, main.py, and the Architectures/ folder.

The folder should include four subfolders:
prj_03, prj_04, prj_05, and prj_06

Each subfolder contains:

~90 simulation output files in HDF5 format — each one is a different flood event over a different terrain
A Terrains/ folder with the terrain TIFF files used in the simulations
