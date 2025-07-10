## Database directory: 
Use a more specific directory like:

./Database/chunks/{prj_num}/
This makes it clear that:

These are partial datasets
Belong to a specific project
Were saved per sublist chunk for memory efficiency

## The project is composed of 3 parts: 
1. preprocessing: Convert raw HEC-RAS simulation results (in HDF5 and terrain files) into training-ready datasets
2. training DL models
3. closure part


from_HDF script since it's the first step in your preprocessing pipeline

## Data Availability
The folder `HECRAS_Simulations_Results/` contains raw terrain and simulation files generated using HEC-RAS. Due to GitHub file size limitations, these files are **not included** in this repository. You can download the full dataset from the following OneDrive link:  
[Download Simulations Results](https://onedrive.live.com/your_shared_link_here)

This dataset is the **starting point of the project**. All subsequent steps in the pipeline depend on these simulations. The project begins by processing these HEC-RAS simulation results and preprocessing them for use in Deep Learning tasks, and finally in the closure model. 

After downloading, place the folder named `HECRAS_Simulations_Results` in the **main directory** of the cloned repository — that is, in the same location as `train.py`, `main.py`, and the `Architectures/` folder. The structure must be preserved as-is to ensure compatibility with the provided data loaders and training scripts.

The folder should contain four subfolders: `prj_03`, `prj_04`, `prj_05`, and `prj_06`.  
Each of these subfolders includes:
- Approximately 90 simulation output files in HDF5 format, each representing a different flood event occuring on a different terran
- A `Terrains/` folder containing the terrain files used in the simulations

