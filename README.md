## Data Availability

The folder `HECRAS_Simulations_Results/` contains raw terrain and simulation files generated using HEC-RAS. Due to GitHub file size limitations, these files are **not included** in this repository. You can download the full dataset from the following OneDrive link:  
[Download Simulations Results](https://onedrive.live.com/your_shared_link_here)

This dataset is the **starting point of the project**. All subsequent steps in the pipeline depend on these simulations. The project begins by processing these HEC-RAS simulation results and preparing them for use in machine learning tasks.

After downloading, place the folder named `HECRAS_Simulations_Results` in the **main directory** of the cloned repository — that is, in the same location as `train.py`, `main.py`, and the `Architectures/` folder.

The folder should contain four subfolders: `prj_03`, `prj_04`, `prj_05`, and `prj_06`.  
Each of these subfolders includes:
- A `Terrains/` folder containing the terrain files used in the simulations
- Approximately 90 simulation output files in HDF5 format, each representing a different flood event or configuration

The structure must be preserved as-is to ensure compatibility with the provided data loaders and training scripts.
