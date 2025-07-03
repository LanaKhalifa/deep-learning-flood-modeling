
# Closure_Part – Patch-Based Closure Model for Flood Simulation

This module applies a trained deep learning model (patch-based) to predict future water depths over large domains simulated by HEC-RAS. It uses an **iterative closure strategy** to handle unknown internal boundary conditions.

---

## 🔍 What It Does

- Extracts HEC-RAS depth vectors and terrain TIFF data.
- Constructs overlapping patches over the large domain.
- Applies a trained neural network (e.g., UNet-based) to predict changes in water depth.
- Iteratively updates the water depth field until convergence.
- Evaluates results using **L1** and **RAE** metrics.
- Produces plots to visualize convergence and prediction quality.

---

## 🧩 Folder Contents

| File                  | Description |
|-----------------------|-------------|
| `main.py`             | Entry point. Runs closure loop over multiple simulations and timestamps. |
| `deep_closure.py`     | The `DeepClosure` class. Orchestrates data loading, inference, reconstruction, and plotting. |
| `config.py`           | Central configuration: patch size, paths, constants. |
| `hecras_loader.py`    | Loads water depth and cell coordinates from HEC-RAS `.hdf` files. |
| `tiff_processor.py`   | Loads terrain elevation data from `.tif` files. |
| `patch_utils.py`      | Slices domain into patches for input to the model. |
| `model_utils.py`      | Loads the trained model and prepares tensors for inference. |
| `plot_utils.py`       | Visualizes convergence metrics and comparison maps (9-panel plot). |

---

## 🧠 Algorithm Summary

> Given known boundary conditions at the edge of a domain, and **initial water depths as a first guess** for the interior, the model:
> 1. Predicts the change in water depth for all patches.
> 2. Updates interior values while keeping boundaries fixed.
> 3. Repeats this process with shifted patch configurations (A, B, C, D).
> 4. Stops when the change between iterations falls below a tolerance.

---

## 📈 Output

- Iterative convergence plots (MAE).
- Visual comparison of predicted vs. ground truth matrices.
- Saved evaluation metrics: `L1` and `RAE` for each simulation.

---

## 🧪 Requirements

- PyTorch
- NumPy
- h5py
- tifffile
- scikit-learn
- matplotlib
