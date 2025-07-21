
import matplotlib.pyplot as plt
import numpy as np

# Function to remove outliers using the Interquartile Range (IQR) method
def remove_outliers_iqr(RAE, num_cells):
    q1 = np.percentile(RAE, 25)
    q3 = np.percentile(RAE, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # Convert num_cells to a NumPy array for consistent indexing
    num_cells = np.array(num_cells)
    return RAE[(RAE >= lower_bound) & (RAE <= upper_bound)], num_cells[(RAE >= lower_bound) & (RAE <= upper_bound)]

# Define the project number and paths to load data
prj_num = '05'  # Replace with your actual project number
rae_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure/Closure_Loop/L1_train_val/prj_{prj_num}_RAE_train_val.npy'
num_cells_path = f'/home/lana_k/Spyder_Projects/Inspect_HDF/Inspect_HDF_thesis_final/Closure/Closure_Loop/L1_train_val/prj_{prj_num}_num_cells.npy'

# Load the arrays of RAE values and number of cells
RAE_train_val = np.load(rae_path)
num_cells = np.load(num_cells_path)

# Flatten the list of lists to a single list and convert to NumPy array
num_cells = np.array([item for sublist in num_cells for item in sublist])
num_cells = num_cells[:len(RAE_train_val)]

# Remove outliers from the RAE values using the IQR method
RAE_train_val_filtered, sizes = remove_outliers_iqr(RAE_train_val, num_cells)

# Group the data into 4 groups: t0, t1, t2, t3
RAE_t0 = RAE_train_val_filtered[0::4]
RAE_t1 = RAE_train_val_filtered[1::4]
RAE_t2 = RAE_train_val_filtered[2::4]
RAE_t3 = RAE_train_val_filtered[3::4]

sizes_t0 = sizes[0::4]
sizes_t1 = sizes[1::4]
sizes_t2 = sizes[2::4]
sizes_t3 = sizes[3::4]


# Define x-axis values for each group to avoid overlap
x_t0 = [1] * len(RAE_t0)
x_t1 = [2] * len(RAE_t1)
x_t2 = [3] * len(RAE_t2)
x_t3 = [4] * len(RAE_t3)


# Define x-axis values for each group, allowing slight scatter along the x-axis
x_t0 = np.random.normal(1, 0.5, len(RAE_t0))  # Centered around 1 with small scatter
x_t1 = np.random.normal(2, 0.5, len(RAE_t1))  # Centered around 2 with small scatter
x_t2 = np.random.normal(3, 0.5, len(RAE_t2))  # Centered around 3 with small scatter
x_t3 = np.random.normal(4, 0.5, len(RAE_t3))  # Centered around 4 with small scatter

# Create scatter plot with specified colors and sizes
plt.figure(figsize=(90, 50))
plt.scatter(x_t0, RAE_t0, s=sizes_t0/20, color='indianred', label='RAE_t0', alpha=0.7)
plt.scatter(x_t1, RAE_t1, s=sizes_t1/20, color='darkorange', label='RAE_t1', alpha=0.7)
plt.scatter(x_t2, RAE_t2, s=sizes_t2/20, color='seagreen', label='RAE_t2', alpha=0.7)
plt.scatter(x_t3, RAE_t3, s=sizes_t3/20, color='royalblue', label='RAE_t3', alpha=0.7)

# Annotate each point with its size
for i in range(len(RAE_t0)):
    if np.random.normal(10, 5,1) > 18:
        plt.text(x_t0[i], RAE_t0[i], f"{round(sizes_t0[i] / 10000, 2)} km²", fontsize=10, ha='left')

for i in range(len(RAE_t1)):
    if np.random.normal(10, 5, 1) > 18:
        plt.text(x_t1[i], RAE_t1[i], f"{round(sizes_t1[i] / 10000, 2)} km²", fontsize=10, ha='left')

for i in range(len(RAE_t2)):
    if np.random.normal(10, 5, 1) > 18:
        plt.text(x_t2[i], RAE_t2[i], f"{round(sizes_t2[i] / 10000, 2)} km²", fontsize=10, ha='left')

for i in range(len(RAE_t3)):
    if np.random.normal(10, 5, 1) > 18:
        plt.text(x_t3[i], RAE_t3[i], f"{round(sizes_t3[i] / 10000, 2)} km²", fontsize=30, ha='left')

# Customize the plot
plt.xlabel('Groups')
plt.ylabel('RAE Values', fontsize = 100)
plt.grid(True)
plt.ylabel('Relative Absolute Errors (RAE)', fontsize=100)
plt.yticks(fontsize=100)
plt.show()



