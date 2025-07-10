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
Closure: Use the trained models to simulate flood progression over full domains.
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
