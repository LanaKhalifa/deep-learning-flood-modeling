import numpy as np
import copy

def infer_grid_shape(cell_coords: np.ndarray) -> tuple[int, int]:
    """
    Infers number of rows and columns from HECRAS cell center coordinates.

    Args:
        cell_coords (np.ndarray): Array of shape (num_cells, 2)

    Returns:
        tuple[int, int]: (num_rows, num_cols) excluding 1-cell-wide boundary.
    """
    threshold = 1.0

    col_y = cell_coords[0][1]
    row_x = cell_coords[0][0]

    num_cols = sum(abs(row[1] - col_y) < threshold for row in cell_coords) - 2
    num_rows = sum(abs(row[0] - row_x) < threshold for row in cell_coords) - 2

    return num_rows, num_cols

def reshape_depth_vectors(depth_vectors: np.ndarray, t: int, delta_t: int, num_rows: int, num_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshapes 1D depth vectors at time t and t+delta_t to 2D matrices.

    Args:
        depth_vectors (np.ndarray): Array of shape (timesteps, num_cells)
        t (int): Current time index
        delta_t (int): Time step forward
        num_rows (int): Number of rows in domain
        num_cols (int): Number of columns in domain

    Returns:
        tuple[np.ndarray, np.ndarray]: (depth_matrix_t, depth_matrix_t_plus_dt)
    """
    num_cells = num_rows * num_cols
    depth_t = depth_vectors[t][:num_cells]
    depth_t_plus = depth_vectors[t + delta_t][:num_cells]

    depth_matrix = np.reshape(depth_t, (num_rows, num_cols))
    depth_matrix_next = np.reshape(depth_t_plus, (num_rows, num_cols))

    return depth_matrix, depth_matrix_next

def calculate_patch_counts(num_rows: int, num_cols: int, cells_in_patch: int) -> tuple[int, int]:
    """
    Calculates number of full patches in each direction.

    Args:
        num_rows (int): Total number of rows
        num_cols (int): Total number of columns
        cells_in_patch (int): Patch size in cells

    Returns:
        tuple[int, int]: (num_patches_row, num_patches_col)
    """
    patches_row = (num_rows - 1) // cells_in_patch
    patches_col = (num_cols - 1) // cells_in_patch
    return patches_row, patches_col

def trim_to_patches(matrix: np.ndarray, num_rows: int, num_cols: int, cells_in_patch: int) -> np.ndarray:
    """
    Trims matrix to fit full patches only.

    Args:
        matrix (np.ndarray): Input 2D matrix
        num_rows (int): Original number of rows
        num_cols (int): Original number of columns
        cells_in_patch (int): Patch size

    Returns:
        np.ndarray: Trimmed matrix
    """
    new_rows = (num_rows - 1) // cells_in_patch * cells_in_patch
    new_cols = (num_cols - 1) // cells_in_patch * cells_in_patch
    return matrix[:new_rows, :new_cols]

def initialize_internal_bcs(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Sets internal region of matrix from reference values, leaving boundaries.

    Args:
        current (np.ndarray): Matrix to modify
        reference (np.ndarray): Matrix with initial internal values

    Returns:
        np.ndarray: Updated matrix
    """
    updated = current.copy()
    updated[1:-1, 1:-1] = reference[1:-1, 1:-1]
    return updated

def copy_ground_truth(matrix: np.ndarray) -> np.ndarray:
    """
    Creates a deep copy of ground truth matrix.

    Args:
        matrix (np.ndarray): Ground truth

    Returns:
        np.ndarray: Deep copy
    """
    return copy.deepcopy(matrix)
