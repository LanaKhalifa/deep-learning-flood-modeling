Overview of the Databases

We have three main datasets used throughout this project:

big_train_val and big_test:
Created from all simulations.
big_test contains patches from simulations that were not used during training/validation.
prj_03_train_val and prj_03_test:
Specific to Project 03, which was manually curated. These datasets help evaluate performance on a consistent and clean dataset.
small_train_val:
A compact version of the dataset used for quick parameter tuning, to speed up experimentation.
Project Numbers

Whenever you see identifiers like 03, 04, 05, or 06, they refer to the four HEC-RAS projects used in this study.
Each project includes:

Multiple simulations (referred to in HEC-RAS as plans).
Terrain files located nearby in the folder structure.
Database Directory Structure

Preprocessed chunks of data are saved under:

./Database/chunks/{prj_num}/
This structure ensures:

Modularity per project.
Efficient storage of sublist-based data chunks, due to memory constraints during processing.
Project Structure

The repository is organized into three main components:

Preprocessing
Converts raw HEC-RAS simulation results (HDF5 and terrain TIFFs) into patches ready for training.
Training
Deep learning models are trained to emulate hydrodynamic simulation results.
Closure
Final step in the pipeline, which uses the trained models to predict forward in time across a full domain.
Entry Script: from_HDF.py

The from_HDF script is the starting point of the preprocessing pipeline.
It:

Loads raw simulation outputs and terrain files.
Extracts relevant data patches from simulations.
Applies filtering and augmentation.
Saves the resulting data into project-specific subdirectories.
Data Availability

The folder HECRAS_Simulations_Results/ contains the raw HEC-RAS simulation outputs and terrain files.
Due to GitHub’s file size limitations, these files are not included in this repository.

You can download them from the following OneDrive link:
📥 Download Simulation Results

Important:
Once downloaded, place the folder HECRAS_Simulations_Results/ at the root of the cloned repository. This ensures compatibility with all scripts. The directory should sit alongside train.py, main.py, and the Architectures/ folder.

Directory Structure:
HECRAS_Simulations_Results/
│
├── prj_03/
│   ├── *.hdf     # ~90 simulation results
│   └── Terrains/
│       └── *.tif
├── prj_04/
├── prj_05/
└── prj_06/
Each *.hdf file corresponds to a flood simulation under different terrain and conditions. Each Terrains/ folder contains the associated elevation map used in the corresponding simulations.
