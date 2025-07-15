# 🌊 Deep Learning Flood Modeling

This repository presents a complete and modular pipeline for emulating hydrodynamic flood simulations using deep learning. It transforms traditional HEC-RAS model outputs into a format suitable for modern deep learning workflows, enabling rapid and scalable flood prediction.

The pipeline consists of three stages:

1. **From Simulations to Data** – Convert HEC-RAS outputs into structured, augmented patch datasets.
2. **Training and Validation** – Train deep learning models to learn flood dynamics at the patch level.
3. **Closure Model Application** – Build a closure model that translates patch-level predictions into full-domain flood simulations.

---
#### Terminology

- **Simulation** = a **HEC-RAS plan**. These terms are used interchangeably. "Simulation" is more intuitive; "plan" is the term used in HEC-RAS.
- Each **project** (prj_03, prj_04, etc.) contains dozens of simulations, each associated with one terrain.
---













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
